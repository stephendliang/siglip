# SASS: symbol-only first, full dump on disk only

`cuobjdump --dump-sass` on a B200 cubin is tens of thousands of lines. Don't
paste that anywhere near context.

## Workflow

1. Symbol listing — which kernels exist:
       cuobjdump --dump-elf-symbols ./fc2-w3-dgswizzle | grep ' T '

2. Instruction-class grep — what classes dominate:
       cuobjdump --dump-sass ./fc2-w3-dgswizzle \
         | grep -cE 'UTCQMMA|STSM|STS\.|LDTM|UTMASTG|F2FP|BSYNC|MEMBAR'

3. Dump to disk, Read with offset/limit:
       cuobjdump --dump-sass --function '<kernel>' \
           /opt/cuda/lib64/libcublasLt.so.13 > /tmp/rank1.sass
       wc -l /tmp/rank1.sass
       # then: Read /tmp/rank1.sass offset=<hit line> limit=40

## Diffing ours vs rank-1

Never load both full SASS files into context. Filter to instruction lines,
diff those:

    for f in ours rank1; do
      grep -oE '[A-Z][A-Z0-9_]+(\.\w+)*' /tmp/$f.sass \
        | sort | uniq -c | sort -rn > /tmp/$f.ops
    done
    diff /tmp/rank1.ops /tmp/ours.ops
