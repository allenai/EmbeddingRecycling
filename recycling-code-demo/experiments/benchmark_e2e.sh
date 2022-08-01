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
START_DT="$(date +'%Y-%m-%d_%H-%M')"


###### CONFIGURATION HERE ######
tries=7
backbones=(
    'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large'
    'nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large'
    'bert-base-uncased'
    'bert-large-uncased'
)
half_precision='true'
batch_size='128'
fetch_ahead='16'
# fetch_spawn='thread'
fetch_spawn='process'
cache_path='/tmp/r3'
###############################


for backbone in "${backbones[@]}"; do

    # start by removing existing cache
    rm -rf ${cache_path}

    # create location for logs
    mkdir -p "${HOME}/benchmarks"

    set -x

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
            cache.half_precision=${half_precision} \
            cache.path=${cache_path} \
            logs_path="${HOME}/benchmarks/r3_e2e-${START_DT}.log"

    set +x

    for try in $(seq 1 ${tries}); do
        echo ""
        echo ""
        printf "Running benchmark for backbone %s\n" "${backbone}/${try}"
        echo ""

        set -x # print commands

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
            cache.half_precision=${half_precision} \
            fetch_spawn=${fetch_spawn} \
            cache.path=${cache_path} \
            logs_path="${HOME}/benchmarks/r3_e2e-${START_DT}.log"

        set +x # don't print commands

        sleep 3
        echo ""
        echo ""
    done
done
