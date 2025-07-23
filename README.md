# Porosity-Permeability Empirical Correlation Streamlit App

## 🌟 Project Overview

This Streamlit web application allows users to explore the empirical relationship between porosity and permeability in geological formations. Utilizing log-linear regression, the app derives a best-fit correlation equation and visualizes the relationship on a semi-log plot, which is typical for this type of petrophysical data. Users can upload their own CSV datasets containing porosity and permeability measurements for analysis.

---

## ✨ Features

* **CSV Data Upload**: Easily upload your own comma-separated values (CSV) files.
* **Dynamic Column Selection**: Select the appropriate porosity and permeability columns from your uploaded data.
* **Log-Linear Regression Analysis**: Automatically perform linear regression on Porosity vs. ln(Permeability).
* **Key Metrics Display**: View the calculated regression coefficients (slope and intercept) and the R-squared value for the log-linear fit.
* **Clear Empirical Correlation Equation**: Displayed in both log-linear and exponential forms using LaTeX.
* **Interactive Semi-Log Plot**: Scatter plot of Porosity vs. Permeability with logarithmic y-axis and regression line.
* **Permeability Prediction**: Enter porosity to get real-time permeability predictions.

---

## 🚀 How to Use

### 1. Deploy on Streamlit Community Cloud (Recommended)

**Preparation:**
Ensure you have a `requirements.txt` file with:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

**Steps:**

1. Push your `app.py`, `requirements.txt`, and (optionally) `porosity_permeability_dataset.csv` to GitHub.
2. Visit [Streamlit Cloud](https://streamlit.io/cloud), log in with GitHub, and deploy your app.
3. Select repository, branch (e.g., `main`), and file path (`app.py`).
4. Click **Deploy!**

**Live App Link:**
[https://rharsh9162--porosity-permeability-relationship-analy-app-xfspcf.streamlit.app/](https://rharsh9162--porosity-permeability-relationship-analy-app-xfspcf.streamlit.app/)

---

### 2. Run Locally

**Clone Repository:**

```bash
git clone https://github.com/rharsh9162/-Porosity-Permeability-Relationship-Analysis.git
cd -Porosity-Permeability-Relationship-Analysis
```

**Create Virtual Environment:**

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

**Run the App:**

```bash
streamlit run app.py
```

---

## 📊 Data

The app expects a CSV file with at least two columns:

* **Porosity**: Preferably in percentage (e.g., 12.5), but fraction format (0.125) will be handled
* **Permeability**: Measured in millidarcies (mD)

An optional script can generate synthetic datasets that mimic real-world porosity-permeability relationships for testing.

---

## 📊 Understanding the Correlation Equation

Permeability in porous rocks depends exponentially on porosity due to:

* **Pore Connectivity**: Small geometric changes in pores lead to large flow differences
* **Fluid Flow Physics**: Kozeny-Carman and similar models suggest exponential relationships
* **Wide Data Range**: Permeability can span several orders of magnitude

To simplify this, we use a **log-linear transformation**:

```
ln(Permeability) = Intercept + Slope × Porosity
```

Which transforms back to:

```
Permeability = e^Intercept × e^(Slope × Porosity)
```

This approach gives a more meaningful and accurate fit than direct linear regression.

---

## 📁 Project Structure

```
.
├── app.py                        # Main Streamlit app
├── requirements.txt              # Dependencies
├── porosity_permeability_dataset.csv  # Example data
└── README.md                     # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Submit issues
* Fork and improve
* Suggest new features or analysis techniques

---

## 📄 License

MIT License — use freely with attribution.

---

## ✉️ Contact

For questions or feedback, please reach out via GitHub issues or email.

---

**Happy Exploring!** 🚀
