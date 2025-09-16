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
  gh label create $l.name --color $l.color --description $l.description 2>$null
  if ($LASTEXITCODE -ne 0) {
    gh label edit $l.name --color $l.color --description $l.description | Out-Null
  }
}

# Sync milestones
$milestones = Get-Content -Raw -Path "build/gh/milestones.json" | ConvertFrom-Json
foreach ($m in $milestones) {
  gh api "/repos/$Repo/milestones" --method POST -H "Accept: application/vnd.github+json" -f title="$($m.title)" -f state="$($m.state)" -f description="$($m.description)" 2>$null
  if ($LASTEXITCODE -ne 0) {
    $existing = gh api "/repos/$Repo/milestones" --paginate | ConvertFrom-Json | Where-Object { $_.title -eq $m.title }
    if ($existing) {
      gh api "/repos/$Repo/milestones/$($existing.number)" --method PATCH -f state="$($m.state)" -f description="$($m.description)" | Out-Null
    }
  }
}

# Issues (idempotent)
$existingIssues = gh issue list --limit 1000 --json title,number | ConvertFrom-Json
Get-Content "build/gh/issues.jsonl" | ForEach-Object {
  $obj = $_ | ConvertFrom-Json
  $found = $existingIssues | Where-Object { $_.title -eq $obj.title }
  $labelArgs = @(); foreach ($lab in $obj.labels) { $labelArgs += @("--label", $lab) }
  $msArg = @(); if ($obj.milestone) { $msArg = @("--milestone", $obj.milestone) }
  if ($found) {
    # Update labels and milestone on existing issue
    gh issue edit $found.number @labelArgs @msArg | Out-Null
  } else {
    gh issue create --title $obj.title --body $obj.body @labelArgs @msArg | Out-Null
  }
}

Write-Output "Sync complete."
