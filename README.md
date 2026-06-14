# Movie Rating Prediction API

A Flask web application that predicts IMDb-style movie ratings (1–10) based on movie metadata, using a pre-trained machine learning model.

This is **Part 3** of the Machine Learning final assignment — wrapping the model from Part 2 in a web service.

---


## Prerequisites

* **Python 3.10 – 3.12**
* `pip` (comes bundled with Python)
* `git`

---

## Installation

**1. Clone the repository and navigate into it:**

```bash
git clone https://github.com/BarelAndArad/Prediction_App_Flask.git
cd Prediction_App_Flask
```

**2. Create and activate a virtual environment:**

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

**3. Install the dependencies:**

```bash
pip install -r requirements.txt
```

---

## Running the Server

From the project root (with the virtual environment activated):

```bash
python api.py
```

The server starts on:

**[http://localhost:5000](http://localhost:5000)**

Open that URL in your browser to use the prediction form.
To stop the server, press `Ctrl + C` in the terminal.

---

## Input Fields

The HTML form requires the following fields. All fields must be filled before submitting.

| Field | Type | Expected Range / Format | Example |
|---|---|---|---|
| `primaryTitle` | text | Any non-empty string | `The Shawshank Redemption` |
| `startYear` | number | Integer between **1888** and the current year | `1994` |
| `runtimeMinutes` | number | Positive number (in minutes) | `142` |
| `num_lead_actors` | number | Non-negative integer | `2` |
| `Language` | text | Movie language | `English` |
| `Country` | text | Country of production | `United States` |
| `plot` | text | Short plot summary | `Two imprisoned men bond over years...` |
| `genres` | text | Comma-separated genres | `Drama` or `Drama,Thriller` |

The predicted rating is returned as a number between **1 and 10** (rounded to one decimal place).

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Returns the HTML form (`index.html`) |
| `/predict` | `POST` | Accepts JSON with movie fields, returns `{"predicted_rating": <value>}` |

### Error handling

| Situation | HTTP Status | Response |
|---|---|---|
| Missing field | `400` | `{"error": "status 400 : Oops! Looks like <field> is missing."}` |
| Non-numeric where number expected | `400` | `{"error": "status 400 : ..."}` |
| Out-of-range value (year < 1888, negative runtime, etc.) | `400` | `{"error": "status 400 : ..."}` |
| Unexpected internal error | `500` | `{"error": "<exception message>"}` |

---

## Project Structure

```
Prediction_App_Flask/
├── api.py                  # Flask backend — endpoints and prediction logic
├── assets_data_prep.py     # prepare_data() function from Part 2 (unchanged)
├── trained_model.pkl       # Trained model from Part 2 (serialized with pickle)
├── requirements.txt        # All dependencies with versions (pip freeze style)
├── README.md               # This file
├── .gitignore              # Ignored files (venv, __pycache__, ...)
└── templates/
    └── index.html          # Frontend form and result display
```

---

## Self-Check Before Submission

To make sure everything works on a clean machine, run this end-to-end check:

```bash
git clone https://github.com/BarelAndArad/Prediction_App_Flask.git fresh_clone
cd fresh_clone
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
python api.py
```

Then in a browser:

1. Open [http://localhost:5000](http://localhost:5000)
2. Fill in the form with valid data
3. Click **Predict Rating**
4. Verify a numeric rating appears on the page