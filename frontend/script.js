document.getElementById("strokeForm").onsubmit = async function(e){

e.preventDefault()

const data = {

age: parseFloat(document.getElementById("age").value),
avg_glucose_level: parseFloat(document.getElementById("glucose").value),
bmi: parseFloat(document.getElementById("bmi").value),
hypertension: parseInt(document.getElementById("hypertension").value),
heart_disease: parseInt(document.getElementById("heart_disease").value)

}

const resultBox = document.getElementById("result")

// prediction
const response = await fetch("http://127.0.0.1:8000/predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(data)
})

const result = await response.json()

const risk = (result.stroke_probability*100).toFixed(2)

resultBox.innerHTML = `Stroke Risk: ${risk}%`

// update risk gauge
document.getElementById("riskGauge").value = risk


// local SHAP
const shapRes = await fetch("http://127.0.0.1:8000/explain",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(data)
})

const shapData = await shapRes.json()

document.getElementById("shapPlot").src =
"data:image/png;base64,"+shapData.plot

// force plot
const forceRes = await fetch("http://127.0.0.1:8000/force_plot",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(data)
})

const forceData = await forceRes.json()

const iframe = document.getElementById("forceFrame")

iframe.srcdoc = forceData.plot

// global SHAP summary
const summaryRes = await fetch("http://127.0.0.1:8000/summary_plot")

const summaryData = await summaryRes.json()

document.getElementById("summaryPlot").src =
"data:image/png;base64,"+summaryData.plot


// feature importance
const importanceRes = await fetch("http://127.0.0.1:8000/feature_importance")

const importanceData = await importanceRes.json()

document.getElementById("importancePlot").src =
"data:image/png;base64," + importanceData.plot

}


