process PREPARE_MODEL_CONFIG {
    input:
    val(meta)

    output:
    tuple val(meta), path("model.config"), emit: config

    script:
    def header = meta.keySet().join(",")
    def values = meta.values().join(",")
    """
    #!/bin/bash
    # Create a vertical CSV file with the meta data
    echo "$header" > tmp
    echo "$values" >> tmp

    awk -F',' '
    {
        for (i = 1; i <= NF; i++)  {
            a[NR,i] = \$i
        }
    }
    NF > p { p = NF }
    END {
        for(j = 1; j <= p; j++) {
            str = a[1,j]
            for(i = 2; i <= NR; i++){
                str = str "," a[i,j]
            }
            print str
        }
    }
    ' tmp > model.config

    """
}