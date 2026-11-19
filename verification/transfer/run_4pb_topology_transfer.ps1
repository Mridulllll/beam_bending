param(
    [string]$Repo = (Get-Location).Path,
    [string[]]$TopologiesToRun = @("Primitive", "Diamond"),
    [int[]]$NShearToRun = @(6, 24),
    [double]$RelativeDensity = 0.12,
    [string]$CampaignType = "sizing_shape",
    [string]$LoadCase = "compression",
    [string]$State = "initial",
    [int]$NumCpus = 8
)

$ErrorActionPreference = "Stop"

$Repo = (Resolve-Path $Repo).Path
Set-Location $Repo

$Builder = Join-Path $Repo "fem\abaqus\builders\build_4pb_homogenized_beam.py"
$Postprocessor = Join-Path $Repo "fem\abaqus\postprocessing\postprocess_4pb_homogenized.py"
$Validator = Join-Path $Repo "fem\abaqus\postprocessing\validate_4pb_post.py"
$Preflight = Join-Path $Repo "verification\transfer\preflight_4pb_topology_transfer.py"
$Analyzer = Join-Path $Repo "verification\transfer\analyze_4pb_topology_transfer.py"
$SrcPath = Join-Path $Repo "src"

$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$PathsFile = Join-Path $Repo "config\paths.local.json"
if (-not (Test-Path $PathsFile)) {
    throw "Missing path configuration: $PathsFile"
}

$PathConfig = Get-Content $PathsFile -Raw | ConvertFrom-Json
$RunRoot = Join-Path $PathConfig.abaqus_scratch_root "4pb_homogenized"

$EnvironmentNames = @(
    "HOST_4PB_CORE_CAMPAIGN_TYPE",
    "HOST_4PB_CORE_TOPOLOGY",
    "HOST_4PB_CORE_RELATIVE_DENSITY",
    "HOST_4PB_CORE_LOAD_CASE",
    "HOST_4PB_CORE_STATE",
    "HOST_4PB_N_SHEAR",
    "HOST_4PB_NE_CELL",
    "HOST_4PB_FACE_NE_Z",
    "HOST_4PB_ELEMENT_TYPE",
    "HOST_4PB_FIXTURE_MODE",
    "HOST_4PB_DOMAIN_MODE",
    "HOST_4PB_SUBMIT_JOB",
    "HOST_4PB_NUM_CPUS",
    "PYTHONPATH"
)

$SavedEnvironment = @{}

foreach ($Name in $EnvironmentNames) {
    $Item = Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue

    if ($null -eq $Item) {
        $SavedEnvironment[$Name] = $null
    }
    else {
        $SavedEnvironment[$Name] = $Item.Value
    }
}


function Restore-HostEnvironment {
    foreach ($Name in $EnvironmentNames) {
        $Value = $SavedEnvironment[$Name]

        if ($null -eq $Value) {
            Remove-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
        }
        else {
            Set-Item -Path "Env:$Name" -Value $Value
        }
    }
}


function Read-JsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (Get-Content $Path -Raw | ConvertFrom-Json)
}


function Test-DoubleClose {
    param(
        [double]$A,
        [double]$B,
        [double]$Tolerance = 1e-9
    )

    return ([math]::Abs($A - $B) -le $Tolerance)
}


function Find-TransferCase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Topology,

        [Parameter(Mandatory = $true)]
        [int]$NShear
    )

    if (-not (Test-Path $RunRoot)) {
        return $null
    }

    $Matches = @()

    foreach ($Directory in (Get-ChildItem $RunRoot -Directory)) {
        $ManifestPath = Join-Path $Directory.FullName "manifest.json"

        if (-not (Test-Path $ManifestPath)) {
            continue
        }

        try {
            $Manifest = Read-JsonFile -Path $ManifestPath
        }
        catch {
            continue
        }

        if ([int]$Manifest.model_implementation_revision -ne 6) {
            continue
        }

        if ($Manifest.database_query.topology -ne $Topology) {
            continue
        }

        $RdMatch = Test-DoubleClose `
            -A ([double]$Manifest.database_query.relative_density_target) `
            -B $RelativeDensity

        if (-not $RdMatch) {
            continue
        }

        if ($Manifest.database_query.campaign_type -ne $CampaignType) {
            continue
        }

        if ($Manifest.database_query.optimization_load_case -ne $LoadCase) {
            continue
        }

        if ($Manifest.database_query.requested_state -ne $State) {
            continue
        }

        if ([int]$Manifest.geometry.n_shear -ne $NShear) {
            continue
        }

        if ($Manifest.mesh.element_type -ne "C3D20R") {
            continue
        }

        if ([int]$Manifest.mesh.core_elements_per_cell_edge -ne 2) {
            continue
        }

        if ([int]$Manifest.mesh.face_elements_through_thickness -ne 2) {
            continue
        }

        if ($Manifest.fixture.mode -ne "section_distributing") {
            continue
        }

        if ($Manifest.domain.mode -ne "quarter_xy") {
            continue
        }

        $Matches += $Directory
    }

    if ($Matches.Count -eq 0) {
        return $null
    }

    return ($Matches | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
}


function Assert-PostQaPass {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CasePath
    )

    $SummaryPath = Join-Path $CasePath "post\summary.json"

    if (-not (Test-Path $SummaryPath)) {
        throw "Missing postprocessing summary: $SummaryPath"
    }

    $Summary = Read-JsonFile -Path $SummaryPath

    if ($Summary.overall_status -ne "PASS") {
        throw "Postprocessing QA is not PASS: $CasePath"
    }

    return $Summary
}


function Postprocess-And-Validate {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.DirectoryInfo]$Case
    )

    Write-Host ""
    Write-Host "Postprocessing:"
    Write-Host "  $($Case.FullName)"

    & abaqus python $Postprocessor `
        --case $Case.FullName `
        --equilibrium-rel-tol 5e-6

    if ($LASTEXITCODE -ne 0) {
        throw "Abaqus postprocessing failed: $($Case.FullName)"
    }

    $Summary = Assert-PostQaPass -CasePath $Case.FullName

    & $Python $Validator --case $Case.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "Validation script failed: $($Case.FullName)"
    }

    $UMid = [double]$Summary.midspan.U3_midplane_avg_mm

    Write-Host ("QA PASS; U3_mid = {0:G12} mm" -f $UMid)
}


try {
    $OldPythonPath = $env:PYTHONPATH

    if ([string]::IsNullOrWhiteSpace($OldPythonPath)) {
        $env:PYTHONPATH = $SrcPath
    }
    else {
        $env:PYTHONPATH = $SrcPath + ";" + $OldPythonPath
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "HOST TOPOLOGY TRANSFER PREFLIGHT"
    Write-Host "============================================================"

    & $Python $Preflight `
        --repo $Repo `
        --relative-density "$RelativeDensity" `
        --campaign-type $CampaignType `
        --load-case $LoadCase `
        --state $State

    if ($LASTEXITCODE -ne 0) {
        throw "Topology-transfer preflight failed."
    }

    $env:HOST_4PB_CORE_CAMPAIGN_TYPE = $CampaignType
    $env:HOST_4PB_CORE_RELATIVE_DENSITY = "$RelativeDensity"
    $env:HOST_4PB_CORE_LOAD_CASE = $LoadCase
    $env:HOST_4PB_CORE_STATE = $State

    $env:HOST_4PB_NE_CELL = "2"
    $env:HOST_4PB_FACE_NE_Z = "2"
    $env:HOST_4PB_ELEMENT_TYPE = "C3D20R"
    $env:HOST_4PB_FIXTURE_MODE = "section_distributing"
    $env:HOST_4PB_DOMAIN_MODE = "quarter_xy"
    $env:HOST_4PB_SUBMIT_JOB = "1"
    $env:HOST_4PB_NUM_CPUS = "$NumCpus"

    foreach ($Topology in $TopologiesToRun) {
        $env:HOST_4PB_CORE_TOPOLOGY = $Topology

        foreach ($NShear in $NShearToRun) {
            $env:HOST_4PB_N_SHEAR = "$NShear"

            Write-Host ""
            Write-Host "============================================================"
            Write-Host ("{0} RD{1:P0} initial   N_A={2}" -f $Topology, $RelativeDensity, $NShear)
            Write-Host "============================================================"

            $Case = Find-TransferCase -Topology $Topology -NShear $NShear

            if ($null -ne $Case) {
                $SummaryPath = Join-Path $Case.FullName "post\summary.json"

                if (Test-Path $SummaryPath) {
                    $ExistingSummary = Read-JsonFile -Path $SummaryPath

                    if ($ExistingSummary.overall_status -eq "PASS") {
                        Write-Host "Reusing existing QA-PASS case:"
                        Write-Host "  $($Case.FullName)"
                        continue
                    }
                }

                $ManifestPath = Join-Path $Case.FullName "manifest.json"
                $Manifest = Read-JsonFile -Path $ManifestPath
                $OdbPath = Join-Path $Case.FullName ($Manifest.job_name + ".odb")

                if (Test-Path $OdbPath) {
                    Write-Host "Existing solved case found; postprocessing it."
                    Postprocess-And-Validate -Case $Case
                    continue
                }

                throw "Matching deterministic case exists but has neither QA-PASS postprocessing nor an ODB: $($Case.FullName)"
            }

            Write-Host "Building and solving new case..."

            & abaqus cae "noGUI=$Builder"

            if ($LASTEXITCODE -ne 0) {
                throw "Abaqus builder failed: $Topology N_A=$NShear"
            }

            $Case = Find-TransferCase -Topology $Topology -NShear $NShear

            if ($null -eq $Case) {
                throw "Builder completed but expected case could not be located: $Topology N_A=$NShear"
            }

            $ManifestPath = Join-Path $Case.FullName "manifest.json"
            $Manifest = Read-JsonFile -Path $ManifestPath

            if ($Manifest.database_query.topology -ne $Topology) {
                throw "Material selection mismatch: requested $Topology but manifest reports $($Manifest.database_query.topology)"
            }

            $ReflectionX = [double]$Manifest.domain.constitutive_reflection_relative_error_x
            $ReflectionY = [double]$Manifest.domain.constitutive_reflection_relative_error_y

            Write-Host "Generated case:"
            Write-Host "  $($Case.FullName)"
            Write-Host ("  reflection x/y: {0:E5} / {1:E5}" -f $ReflectionX, $ReflectionY)

            Postprocess-And-Validate -Case $Case
        }
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "SIX-CASE HOST TRANSFER ANALYSIS"
    Write-Host "============================================================"

    & $Python $Analyzer --run-root $RunRoot

    if ($LASTEXITCODE -ne 0) {
        throw "Topology-transfer analyzer failed."
    }
}
finally {
    Restore-HostEnvironment
    Set-Location $Repo
}
