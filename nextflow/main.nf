#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Stage colormaps
colormaps = file("$workflow.projectDir/bin/colormaps/", checkIfExists: true)

process make_miniature {
    label "process_high"
    publishDir "$params.outdir/$dimred/$metric/$scaler/${n_components}d/$colormap/$log_arg/", mode: 'copy', overwrite: true

    input:
        tuple path(filename), val(dimred), val(metric), val(log_arg), val(n_components), val(colormap), val(scaler)
        path(colormaps)

    output:
        path '*.png'
        tuple path('*.h5'), val(dimred), val(metric), val(log_arg), val(n_components), val(colormap), val(scaler)

    script:
    """
    paint_miniature.py $filename ${filename.simpleName}.png \\
        --dimred $dimred \\
        --n_components $n_components \\
        --colormap $colormap \\
        --metric $metric \\
        --scaler $scaler \\
        --plot_embedding \\
        $log_arg \\
        --level -1 \\
        --save_data \\
        --max_pixels $params.max_size
    """

    stub:
    """
    touch miniature.png
    touch miniature.h5
    """
}

process calc_metrics {
    label "process_high"
    publishDir "$params.outdir/$dimred/$metric/$scaler/${n_components}d/$colormap/$log_arg/", mode: 'copy', overwrite: true

    input:
        tuple path(h5), val(dimred), val(metric), val(log_arg), val(n_components), val(colormap), val(scaler)

    output:
        path 'metrics.h5'

    script:
    """
    miniature_metrics.py $h5 --metric $metric --n $params.n
    """

    stub:
    """
    touch metrics.h5
    """
}

workflow {
    // Read samplesheet and create input channel
    input = Channel.fromPath(params.samplesheet, checkIfExists: true)
        | splitCsv(header: true)
        | map { row ->
            tuple(
                file(row.filename),
                row.dimred,
                row.metric,
                row.log_arg,
                row.components,
                row.colormap,
                row.scaler
            )
        }

    // Run miniature generation
    make_miniature(input, colormaps)

    // Calculate metrics
    make_miniature.out[1] | calc_metrics
}
