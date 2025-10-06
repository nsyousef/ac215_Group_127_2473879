# ac215_Group_127_2473879

## File Structure

`eda`: stores exploratory data analysis for our project

## Temp Notes:

Suggested repo structure (from ChatGPT):

```
your-ml-project/
│
├── eda/                    # EDA code: notebooks, analysis scripts, EDA README
│   ├── eda_images.ipynb
│   └── requirements.txt    # EDA-specific requirements
│
├── app/                    # Application code (to be containerized)
│   ├── service1/           # Each microservice in its own folder
│   │   ├── Dockerfile
│   │   └── ...
│   ├── service2/
│   │   ├── Dockerfile
│   │   └── ...
│   ├── shared/             # Shared code/utilities among microservices
│   └── requirements.txt    # App-wide requirements for building images
│
├── data/                   # Only local (small) lookup tables, not raw data!
│   └── .gitkeep
│
├── scripts/                # Utility scripts (e.g., data downloaders)
│
├── .gitignore
├── README.md               # Project overview, pointers to EDA + app
├── docker-compose.yml      # (optional) For orchestrating services locally
└── environment.yml         # (optional/global) to document top-level deps
```

