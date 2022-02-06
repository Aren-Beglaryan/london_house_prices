# london_house_prices

The project is about to predict the price of the house

# RUN

+ local

    ```
    run app.py
    use request.json with postman or other tool to do a post request 
    ```

+ docker

    to build an image run
    
    ```bash
    make build
    ```

    to run an application you need to have a 
    modelfile in parent directory and encoder file named `model.pt` and `encoder` respectiverly.
    ```
    make run
    ```
