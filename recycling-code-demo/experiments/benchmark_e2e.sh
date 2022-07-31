# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

export PYTHONPATH="${SCRIPT_DIR}/../src:${PYTHONPATH}"


tries=1
backbones=(
    'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large'
    # 'nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large'
    # 'bert-base-uncased'
    # 'bert-large-uncased'
)
half_precision='false'
batch_size='128'
fetch_ahead='16'
fetch_spawn='thread'
cache_path='/tmp/r3'


for backbone in "${backbones[@]}"; do
    # start by removing existing cache
    rm -rf ${cache_path}

    # step 3 is responsible for creating the cache
    python "${SCRIPT_DIR}/benchmark_e2e.py" \
            backbone="${backbone}" \
            dataset.loader.split=validation \
            cache.backend=leveldb \
            device=cuda \
            keep_cache=True \
            steps='[3]' \
            batch_size=${batch_size} \
            fetch_ahead=${fetch_ahead} \
            half_precision=${half_precision} \
            cache.path=${cache_path} \
            logs_path="${HOME}/benchmark_e2e.log"

    for try in $(seq 1 ${tries}); do
        echo ""
        echo ""
        printf "Running benchmark for backbone %s\n" "${backbone}/${try}"
        echo ""

        # step 2,4,5 are responsible for running with no cache,
        # cache but no prefetch, and cache with prefetch respectively.
        python "${SCRIPT_DIR}/benchmark_e2e.py" \
            backbone="${backbone}" \
            dataset.loader.split=validation \
            cache.backend=leveldb \
            device=cuda \
            keep_cache=True \
            steps='[2,4,5]' \
            batch_size=${batch_size} \
            fetch_ahead=${fetch_ahead} \
            half_precision=${half_precision} \
            fetch_spawn=${fetch_spawn} \
            cache.path=${cache_path} \
            logs_path="${HOME}/benchmark_e2e.log"

        sleep 3
        echo ""
        echo ""
    done
done
