import requests

URA_ACCESS_KEY = "5d7f0435-403d-4477-b3bd-a9cf69cc60dd"
URA_TOKEN_URL  = "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1"

r = requests.get(
    URA_TOKEN_URL,
    headers={"AccessKey": URA_ACCESS_KEY},
    timeout=10
)

print("HTTP Status :", r.status_code)
print("Response    :", repr(r.text))