#requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Python,
    [ValidateSet('build','ask','verify')][string]$Mode = 'verify',
    [string]$Question = 'What exact phone number should I call for the University IT Service Desk?',
    [ValidateSet('auto','vulkan')][string]$Backend = 'auto',
    [string]$VulkanDevice = '1',
    [int]$Port = 11521,
    [int]$TimeoutMinutes = 45
)
$ErrorActionPreference = 'Stop'
function Read-Hardware {
    $ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $raw = (& nvidia-smi --id=0 --query-gpu=temperature.gpu,memory.free --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { throw 'GPU monitoring failed' }
    $values = $raw.Trim().Split(',')
    [pscustomobject]@{utc=(Get-Date).ToUniversalTime().ToString('o');ram_gib=[math]::Round($ram,3);gpu_c=[double]$values[0];vram_mib=[double]$values[1]}
}
function Quote-Argument([string]$value) { '"' + $value.Replace('"','\"') + '"' }
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Use a port between 1024 and 65535' }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw 'The requested port is already in use' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$Ollama = (Get-Command ollama -ErrorAction Stop).Source
$before = Read-Hardware
if ($before.ram_gib -lt 4.5 -or $before.gpu_c -ge 75 -or $before.vram_mib -lt 4500) { throw 'Insufficient memory or high GPU temperature before startup' }
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,6)
$run = Join-Path $PSScriptRoot "runs/$stamp"
$null = New-Item -ItemType Directory -Path $run
$url = "http://127.0.0.1:$Port"
$settings = @{OLLAMA_HOST="127.0.0.1:$Port";OLLAMA_NUM_PARALLEL='1';OLLAMA_MAX_LOADED_MODELS='2';OLLAMA_KEEP_ALIVE='120';OMP_NUM_THREADS='2';MKL_NUM_THREADS='2';OPENBLAS_NUM_THREADS='2';PYTHONUNBUFFERED='1';POLICY_THERMAL_GUARD='1'}
if ($Backend -eq 'vulkan') {
    $settings['OLLAMA_VULKAN']='1'; $settings['OLLAMA_LLM_LIBRARY']='vulkan'
    $settings['CUDA_VISIBLE_DEVICES']='-1'; $settings['GGML_VK_VISIBLE_DEVICES']=$VulkanDevice
}
$previous=@{}
$server=$null
$worker=$null
$children=@{}
$failure=$null
$exitCode=1
$start=Get-Date
try {
    foreach ($key in $settings.Keys) { $previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    $server=Start-Process -FilePath $Ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'server.out.log') -RedirectStandardError (Join-Path $run 'server.err.log')
    $ready=$false
    for ($attempt=0;$attempt -lt 30;$attempt++) {
        if ($server.HasExited) { throw 'The temporary Ollama server exited during startup' }
        try { $null=Invoke-RestMethod "$url/api/version" -NoProxy -TimeoutSec 2; $ready=$true; break } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ready) { throw 'The temporary Ollama server did not become ready' }
    $models=(Invoke-RestMethod "$url/api/tags" -NoProxy -TimeoutSec 10).models.name
    foreach ($name in @('qwen3:0.6b','nomic-embed-text:latest')) { if ($name -notin $models) { throw "Missing local model: $name" } }
    $entry=@{build='rag.py';ask='agent.py';verify='verify_live.py'}[$Mode]
    $arguments=@('-u',(Quote-Argument (Join-Path $PSScriptRoot $entry)))
    if ($Mode -eq 'ask') { $arguments += (Quote-Argument $Question) }
    $arguments += @('--base-url',$url)
    if ($Mode -ne 'build') { $arguments += @('--record',(Quote-Argument (Join-Path $run 'answers.json'))) }
    $worker=Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'worker.out.log') -RedirectStandardError (Join-Path $run 'worker.err.log')
    Write-Output "Started $Mode; server PID $($server.Id); worker PID $($worker.Id); records $run"
    $lastReport=$start
    while (-not $worker.HasExited) {
        if ($server.HasExited) { throw 'The temporary server exited before the worker' }
        foreach ($item in Get-CimInstance Win32_Process -Filter "ParentProcessId=$($server.Id)") {
            $child=Get-Process -Id $item.ProcessId -ErrorAction SilentlyContinue
            if ($child -and $child.StartTime -ge $start) { $children[$child.Id]=$child }
        }
        $sample=Read-Hardware
        $sample | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $run 'hardware.jsonl')
        if ($sample.ram_gib -lt 1.25 -or $sample.gpu_c -ge 80) { throw 'Runtime memory or GPU temperature limit reached' }
        if (((Get-Date)-$start).TotalMinutes -gt $TimeoutMinutes) { throw 'Run exceeded its time limit' }
        if (((Get-Date)-$lastReport).TotalSeconds -ge 30) {
            Write-Output ("Monitoring: free RAM {0:N2} GiB; GPU {1} C" -f $sample.ram_gib,$sample.gpu_c)
            $lastReport=Get-Date
        }
        Start-Sleep -Seconds 2
        $worker.Refresh()
    }
    $worker.WaitForExit()
    $exitCode=$worker.ExitCode
    if ($exitCode -ne 0) { throw "Worker exited with code $exitCode" }
}
catch { $failure=$_.Exception.Message; Write-Output "Stopped: $failure" }
finally {
    foreach ($process in @($worker,$server) + @($children.Values)) {
        if ($process -and -not $process.HasExited) {
            try { $process.Kill($true); $process.WaitForExit() } catch { Write-Warning $_.Exception.Message }
        }
    }
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    @{mode=$Mode;backend=$Backend;started_utc=$start.ToUniversalTime().ToString('o');finished_utc=(Get-Date).ToUniversalTime().ToString('o');exit_code=$exitCode;failure=$failure} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'run.json')
    Write-Output "Temporary processes closed; records $run"
}
if ($failure) { exit 1 }
exit $exitCode
