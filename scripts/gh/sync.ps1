Param(
  [Parameter(Mandatory=$true)][string]$Repo
)

# Ensure gh is installed and authenticated
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  Write-Error "GitHub CLI 'gh' not found. Install from https://cli.github.com/."; exit 1
}

$authStatus = gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Error "Not authenticated. Run: gh auth login"; exit 1
}

# Set repo as default for subsequent commands
$env:GH_REPO = $Repo

# Sync labels
$labels = Get-Content -Raw -Path "build/gh/labels.json" | ConvertFrom-Json
foreach ($l in $labels) {
  gh label create $l.name --color $l.color --description $l.description 1>$null 2>$null
  if ($LASTEXITCODE -ne 0) { gh label edit $l.name --color $l.color --description $l.description 1>$null 2>$null }
}

# Sync milestones
$milestones = Get-Content -Raw -Path "build/gh/milestones.json" | ConvertFrom-Json
foreach ($m in $milestones) {
  gh api "/repos/$Repo/milestones" --method POST -H "Accept: application/vnd.github+json" -f title="$($m.title)" -f state="$($m.state)" -f description="$($m.description)" 1>$null 2>$null
  if ($LASTEXITCODE -ne 0) {
    $existing = gh api "/repos/$Repo/milestones" --paginate 2>$null | ConvertFrom-Json | Where-Object { $_.title -eq $m.title }
    if ($existing) { gh api "/repos/$Repo/milestones/$($existing.number)" --method PATCH -f state="$($m.state)" -f description="$($m.description)" 1>$null 2>$null }
  }
}

# Issues (idempotent)
$existingIssues = gh issue list --limit 1000 --state all --json title,number 2>$null | ConvertFrom-Json
Get-Content "build/gh/issues.jsonl" | ForEach-Object {
  $obj = $_ | ConvertFrom-Json
  $found = $existingIssues | Where-Object { $_.title -eq $obj.title }
  if ($null -ne $found -and $found.Count -gt 0) { $issueNum = $found[0].number } else { $issueNum = $null }

  if ($issueNum) {
    $args = @("issue","edit", $issueNum)
    if ($obj.milestone) { $args += @("-m", $obj.milestone) }
    if ($obj.labels) { foreach ($lab in $obj.labels) { $args += @("--add-label", $lab) } }
    gh @args 1>$null 2>$null
  } else {
    $args = @("issue","create","--title", $obj.title, "--body", $obj.body)
    if ($obj.milestone) { $args += @("--milestone", $obj.milestone) }
    if ($obj.labels) { foreach ($lab in $obj.labels) { $args += @("--label", $lab) } }
    gh @args 1>$null 2>$null
  }
}

Write-Output "Sync complete."
