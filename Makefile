all:
	./annular_duct_modes.py
init:
	pip install -r requirements.txt

test:
	nosetests tests
