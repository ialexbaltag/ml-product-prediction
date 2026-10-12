import joblib
import pandas as pd
 
 
 # Load the saved model
model = joblib.load("model/category_model.pkl")
 
print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")
 
while True:
    title = input("Enter product title:")
    if title.lower() == "exit":
        print("Exiting...")
        break
 
    brand = input("Enter a brand:")
    if brand.lower() == "exit":
        print("Exiting...")
        break
 
    # Compute review length
    length = len(title)
 
    # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "product_title": title,
        "product_brand": brand,
        "product_title_length": length
    }])
 
    # Predict category
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)