PY ?= python
all: data train eval          ## everything except the dashboard
setup:                        ## install dependencies
	$(PY) -m pip install -r requirements.txt
	-$(PY) -m pip install -r requirements-optional.txt
data:                         ## calibrate + generate synthetic PMS + database
	$(PY) scripts/01_calibrate.py && $(PY) scripts/02_generate_pms.py
train:                        ## baselines + causal models + demo tenants
	$(PY) scripts/03_train_models.py && $(PY) scripts/07_pick_demo_tenants.py
eval:                         ## full evaluation report
	$(PY) scripts/04_evaluate.py
sens:                         ## sensitivity to assumptions (~3 min)
	$(PY) scripts/05_sensitivity.py
dashboard:
	streamlit run app/dashboard.py
api:
	uvicorn pmsdi.api:app --app-dir src --reload
test:
	$(PY) -m pytest -q
