Param(
  [Parameter(Mandatory=$true)][string]$OrgRepo,
  [Parameter(Mandatory=$false)][string]$ProjectTitle = "StableGLM Plan"
)

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Write-Error "GitHub CLI 'gh' not found."; exit 1 }
$authStatus = gh auth status 2>$null; if ($LASTEXITCODE -ne 0) { Write-Error "Not authenticated. Run: gh auth login"; exit 1 }

# Create a user project (v2). For org projects, use gh project create --org <org>
$owner = ($OrgRepo.Split('/'))[0]
$repo = ($OrgRepo.Split('/'))[1]

# Try to find or create project
$projId = (gh project list --owner $owner --format json | ConvertFrom-Json | Where-Object { $_.title -eq $ProjectTitle }).id
if (-not $projId) {
  gh project create --owner $owner --title $ProjectTitle --format json | Set-Variable resp
  $projId = ($resp | ConvertFrom-Json).id
}

# Add issues from repo to project
$issues = gh issue list --repo $OrgRepo --state open --limit 200 --json number | ConvertFrom-Json
foreach ($i in $issues) {
  gh project item-add --project-id $projId --url "https://github.com/$OrgRepo/issues/$($i.number)" 1>$null 2>$null
}

Write-Output "Project seeded: $ProjectTitle"
