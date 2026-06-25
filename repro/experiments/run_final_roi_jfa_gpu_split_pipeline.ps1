param(
    [switch]$SkipSimulations,
    [switch]$Compile
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$GpuPython = if ($env:GEOVORONOI_FV_GPU_PYTHON) { $env:GEOVORONOI_FV_GPU_PYTHON } else { "python" }
$PlotPython = if ($env:GEOVORONOI_FV_PLOT_PYTHON) { $env:GEOVORONOI_FV_PLOT_PYTHON } else { "python" }
$Runner = Join-Path $Root "src\geovoronoi_fv\run_main7_flow_production.py"
$OverlayBuilder = Join-Path $Root "repro\experiments\build_roi_jfa_gpu_split_overlay.py"
$CompileScript = $env:LATEX_COMPILE_SCRIPT

$SyntheticOut = Join-Path $Root "onehour_runs\synthetic5_dense_lu_protocol_radius_dt5_n100_gpu_face_split_adaptive_active_list"
$BentheimerOut = Join-Path $Root "onehour_runs\bentheimer_dense_lu_protocol_radius_dt50_n100_stride0_gpu_face_split_adaptive_active_list"
$OverlayOut = Join-Path $Root "paper_packages\current_manuscript_with_roi_jfa_gpu_split_overlay"

& $GpuPython -m py_compile $Runner
& $PlotPython -m py_compile $OverlayBuilder

if (-not $SkipSimulations) {
    & $GpuPython $Runner `
        --out-dir $SyntheticOut `
        --case-filter "orthogonal_duct,skewed_duct,A_thin_wall,B_narrow_throat,C_maze" `
        --stride-indices "2" `
        --skip-wall-calibration `
        --fixed-beta-star 1.25 `
        --reference-n-steps 900 `
        --coarse-n-steps 100 `
        --coarse-dt 5 `
        --coarse-linear-solver dense_lu `
        --roi-jfa-tile "8,8,16" `
        --roi-jfa-active-list-threshold 0.75 `
        --gpu-face-split `
        --skip-zero-area-diagnostic `
        --no-export-data

    & $GpuPython $Runner `
        --out-dir $BentheimerOut `
        --case-filter "bentheimer_sandstone_crop" `
        --stride-indices "0" `
        --skip-wall-calibration `
        --fixed-beta-star 1.25 `
        --reference-n-steps 900 `
        --coarse-n-steps 100 `
        --coarse-dt 50 `
        --coarse-linear-solver dense_lu `
        --roi-jfa-tile "8,8,16" `
        --roi-jfa-active-list-threshold 0.75 `
        --gpu-face-split `
        --skip-zero-area-diagnostic `
        --no-export-data
}

if (Test-Path -LiteralPath $OverlayOut) {
    $paperRoot = (Resolve-Path -LiteralPath (Join-Path $Root "paper_packages")).Path
    $resolvedOverlay = (Resolve-Path -LiteralPath $OverlayOut).Path
    if (-not $resolvedOverlay.StartsWith($paperRoot)) {
        throw "Refusing to remove overlay outside paper_packages: $resolvedOverlay"
    }
    if ((Split-Path $resolvedOverlay -Leaf) -ne "current_manuscript_with_roi_jfa_gpu_split_overlay") {
        throw "Unexpected overlay path: $resolvedOverlay"
    }
    Remove-Item -LiteralPath $resolvedOverlay -Recurse -Force
}

& $PlotPython $OverlayBuilder

if ($Compile) {
    $MainTex = Join-Path $OverlayOut "geovoronoi_fv_cmame_draft.tex"
    $GpuSplitSupplement = Join-Path $OverlayOut "supplementary\supplement_gpu_split_timing.tex"
    & $PlotPython $CompileScript $MainTex --recipe pdflatex
    & $PlotPython $CompileScript $MainTex --recipe pdflatex
    & $PlotPython $CompileScript $GpuSplitSupplement --recipe pdflatex
    & $PlotPython $CompileScript $GpuSplitSupplement --recipe pdflatex
}

