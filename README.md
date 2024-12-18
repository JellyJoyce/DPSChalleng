# DPS Challenge Option 2
- Model 00: SARIMAX -- didn't meet the expectation -- deleted
- Model 1: XGBoost
- Model 2: LSTM -- final model

Deployed on AWS ECS - Fargate

- Terminal
curl -X POST http://3.107.49.18:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"year":2021,"month":1}'

- Powershell
Invoke-RestMethod -Uri "http://3.107.49.18:5000/predict" `
>>     -Headers @{ "Content-Type" = "application/json" } `
>>     -Body '{"year":2021,"month":1}' `
>>     -Method POST