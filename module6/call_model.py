import requests

data = {
    "columns": ["Lot Area", "Gr Liv Area", "Garage Area", "SalePrice", "Bldg Type"],
    "data": [[31770, 1656, 528.0, 215000, "Twnhs"]]
}

model_call = requests.post("http://localhost:5001/invocations", json=data)

print(model_call.status_code)
print(model_call.content)
