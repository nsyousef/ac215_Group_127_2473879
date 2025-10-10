# ac215_Group_127_2473879

## File Structure

Within `src`, each folder is a microservice with its own Docker container. Please see the folders for another README with more details on that folder.

`root_directory`
└── `ac215_Group_127_2473879`
    ├──`eda`: stores exploratory data analysis for our project
    └──`src`: stores the source code for our project
        └──`data-processor`: stores the code for processing our raw data into final final data
    
    TODO: describe each folder in src (we don't have to describe each file in the folders here; that can be done in separate READMEs in each folder)


## Building and Running Docker Images

### Build Image
```bash
docker build -t <image-name> -f ac215_Group_127_2473879/src/<container_folder>/Dockerfile .
```

### Run Container
```bash
docker run --rm -ti <image-name>
```

### Notes
- Run commands from root directory (one level above repo)
- Ensure `secrets/` folder exists in root
