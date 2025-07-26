Write-Host "正在设置开发环境变量..." -ForegroundColor Yellow

$env:ANTHROPIC_BASE_URL="https://api.aicodemirror.com/api/claudecode"
$env:ANTHROPIC_API_KEY="sk-ant-api03-y_zzpfqMDUDwzFu5q4_5COmoLbKEIzUnF5g93k3Hd7AnCBUa3sN94WCMUYifo0hi5B18mzK5DUMoVjr9SdC2xA"
$env:ANTHROPIC_AUTH_TOKEN="sk-ant-api03-y_zzpfqMDUDwzFu5q4_5COmoLbKEIzUnF5g93k3Hd7AnCBUa3sN94WCMUYifo0hi5B18mzK5DUMoVjr9SdC2xA"

Write-Host "环境变量设置完成。" -ForegroundColor Green