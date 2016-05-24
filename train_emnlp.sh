export PATH="$PATH:/usr/local/cuda/bin"
#env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu0,floatX=float32" python -u mscale_model.py --ptb --train --context_dependent > context_dependent_ptb.log 2>&1 &
env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu0,floatX=float32" python -u mscale_model.py --cs --train --context_dependent > context_dependent_cs.log 2>&1 &
env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu1,floatX=float32" python -u mscale_model.py --de --train --context_dependent > context_dependent_de.log 2>&1 &

#env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu0,floatX=float32" python -u mscale_model.py --ptb --train --context_free > context_free_ptb.log 2>&1 &
env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu2,floatX=float32" python -u mscale_model.py --cs --train --context_free > context_free_cs.log 2>&1 &
env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu3,floatX=float32" python -u mscale_model.py --de --train --context_free > context_free_de.log 2>&1 &

#env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu0,floatX=float32" python -u mscale_model.py --ptb --train --baseline > baseline_ptb.log 2>&1 &
#env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu1,floatX=float32" python -u mscale_model.py --cs --train --baseline > baseline_cs.log 2>&1 &
#env THEANO_FLAGS="mode=FAST_RUN,lib.cnmem=1,device=gpu2,floatX=float32" python -u mscale_model.py --de --train --baseline > baseline_de.log 2>&1 &

