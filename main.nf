nextflow.enable.dsl=2

params.samplesheet = 'samplesheet.csv'
params.outdir = 'outputs'
params.colormaps = 'bin/colormaps'

process make_miniature {
    //cpus 2
    //memory '2 GB'
    //conda '/Users/ataylor/miniforge3/envs/miniature'
    
    publishDir "$params.outdir/$dimred/$metric/$scaler/${n_components}d/$colormap/$log_arg/", mode: 'copy'
    input:
        tuple file(filename), val(dimred), val(metric), val(log_arg), val(n_components), val(colormap), val(scaler)
    output:
        path '*.png'
        tuple path('*.h5'), val(dimred), val(metric), val(log_arg), val(n_components), val(colormap), val(scaler)

    script:
    """
    cp -r ${NXF_HOME}/assets/adamjtaylor/miniature/$params.colormaps .
    paint_miniature.py $filename miniature.png --dimred $dimred --n_components $n_components --colormap $colormap --metric $metric --scaler $scaler --plot_embedding $log_arg --level -1 --save_data
    """

    stub:
    """
    touch miniature.png
    """
}

process calc_metrics {
   // cpus 2
    //memory '2 GB'
    //conda '/Users/ataylor/miniforge3/envs/miniature'
    publishDir "$params.outdir/$dimred/$metric/$scaler/${n_components}d/$colormap/$log_arg/", mode: 'copy'
    input:
        tuple file(h5), val(dimred), val(metric), val(log_arg), val(n_components), val(colormap), val(scaler)
    output:
        file 'metrics.h5'

    script:
    """
    miniature_metrics.py $h5 --metric $metric
    """

}

workflow {
    Channel.fromPath(params.samplesheet, checkIfExists: true) \
        | splitCsv(header:true) \
        | map { row -> tuple(file(row.filename), row.dimred, row.metric, row.log_arg, row.components, row.colormap, row.scaler)}
        | make_miniature
    make_miniature.out[1] \
        | calc_metrics
}
