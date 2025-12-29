.PHONY: install run test fmt

install:
	python -m pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest -q
