# Variables
PYTHON = python3
PIP = pip3
IMAGE_NAME = risk-management-bny

# -- Local Development --

.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install --no-deps pygooglenews

.PHONY: run_hourly_scraper
run_hourly_scraper:
	$(PYTHON) src/run_scraper.py


.PHONY: clean
clean:
	rm -rf data/state/*.csv
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# -- Docker / Production --
.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME) .

# Runs the container and mounts your local 'data' folder
# so the CSVs appear on your actual laptop, not just inside the container.
.PHONY: docker-run
docker-run:
	docker run --rm -v $(PWD)/data:/app/data $(IMAGE_NAME)