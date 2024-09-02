.PHONY: all clean test kernel

all:
	xmake f --ascend-npu=true -c -v
	xmake -vD

clean:
	rm -rf build/
	rm -rf .xmake/

kernel-build:
	$(MAKE) -C src/ops/rotary_embedding/ascend build

kernel-clean:
	$(MAKE) -C src/ops/rotary_embedding/ascend clean

test-%:
	python operatorspy/tests/%.py --ascend

test:
	# python operatorspy/tests/matmul.py --ascend
	# python operatorspy/tests/rms_norm.py --ascend
	# python operatorspy/tests/swiglu.py --ascend
	python operatorspy/tests/reform.py --ascend
