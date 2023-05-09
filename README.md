# Naukma_api_project
This project's goal is to create a model that perdicts air raids in different Ukrainian regions and provide API for easy access to it.
The project uses weather forecasts and ISW Reports to predict raids.
## Project stages
- Data collection
- Data Preprocessing + Feature Engineering
- Model Training
- Hyperparameter tuning

## API
To set up the project You need to launch 2 api endpoints.

- `weather_forecast_api.py` is an endpoint used for loading weather forecast data
- `alarm_api.py` - is an endpoint for loading predictions from the model and fetching updated weather dataand ISW reports
Examples on how to use both APIs can be found in postman collection `AirRaidPrediction.postman_collection.json`

## Models
`.pkl` files for all the models with tuned parameters are provided in model folder.

## Deployment
All filess relevant to deployment a located in the `server` folder. To automaticly deploy code from GitHub recommended using Git **sparse checkout** feature:

```
git init
git remote add -f origin <url>

git config core.sparseCheckout true
echo "server/" >> .git/info/sparse-checkout

git pull origin main
```

The main application is Flask app in `alarm_api.py`, recommended to use uWSGI:

```
sudo pip3 install uwsgi
uwsgi --http 0.0.0.0:8000 --wsgi-file alarm_api.py --callable app --processes 4 --threads 2
```

Also adviset to use Supervisor (Supervisord) to run the server. Config and launch commands depend on the Supervisor installation.
