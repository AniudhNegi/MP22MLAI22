import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Multiple Disease Prediction ", page_icon=":hospital:")

# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.h5', 'rb'))

heart_disease_model = pickle.load(open('heart_model.h5','rb'))
breast_cancer_model = pickle.load(open('breast_cancer_model.h5', 'rb'))
kidney_disease_model = pickle.load(open('Kidney_model.h5', 'rb'))
liver_disease_model = pickle.load(open('Liver_model.h5', 'rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction ',
                          
                          ['Home',
                           'Medical Assistant',
                           'Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Breast Cancer Prediction',
                           'Kidney Disease Prediction',
                           'Liver Disease Prediction'],
                          icons=['house','person-circle','activity','heart','gender-female','file-medical','at'],
                          default_index=0)
    
# Home Page
if (selected == 'Home'):
        
    st.title("Information about Diseases")

    st.header("Diabetes")
    st.markdown("Diabetes mellitus refers to a group of diseases that affect how your body uses blood sugar (glucose). Glucose is vital to your health because it's an important source of energy for the cells that make up your muscles and tissues. It's also your brain's main source of fuel. The underlying cause of diabetes varies by type. But, no matter what type of diabetes you have, it can lead to excess sugar in your blood. Too much sugar in your blood can lead to serious health problems.")
    st.subheader("Symptoms")
    st.markdown("- Increased thirst\n- Frequent urination\n- Extreme hunger\n- Unexplained weight loss\n- Blurred vision")

    st.header("Heart/ Cardiovascular Disease")
    st.markdown("Cardiovascular disease (heart disease) refers to a group of diseases that affect the heart and blood vessels of your body. These diseases can affect one or many parts of your heart and /or blood vessels. A person may be symptomatic (physically experience the disease) or be asymptomatic (not feel anything at all).")
    st.subheader("Symptoms")
    st.markdown("- Pounding or racing heart (palpitations)\n- Chest pain\n- Sweating\n- Lightheadedness\n- Shortness of breath")

    st.header("Breast Cancer")
    st.markdown("Breast cancer is a type of cancer that starts in the breast. It can start in one or both breasts. After skin cancer, breast cancer is the most common cancer diagnosed in women in the United States. Breast cancer can occur in both men and women, but it's far more common in women.")
    st.subheader("Symptoms")
    st.markdown("- New lump in the breast or underarm (armpit)\n- Thickening or swelling of part of the breast.\n- Irritation or dimpling of breast skin\n- Redness or flaky skin in the nipple area or the breast\n- Pulling in of the nipple or pain in the nipple area")

    st.header("Chronic Kidney Disease")
    st.markdown("Chronic kidney disease (CKD) means your kidneys are damaged and can’t filter blood the way they should. The disease is called “chronic” because the damage to your kidneys happens slowly over a long period of time. This damage can cause wastes to build up in your body. CKD can also cause other health problems.")
    st.subheader("Symptoms")
    st.markdown("- Nausea\n- Vomiting\n- Fatigue and weakness\n- Muscle twitches and cramps\n- Loss of appetite")
    
    st.header("Liver Disease")
    st.markdown("")
    st.subheader("Symptoms")
    st.markdown("- Skin and eyes that appear yellowish (jaundice)\n- Abdominal pain and swelling\n- Swelling in the legs and ankles\n- Itchy skin\n- Dark urine color\n- Pale stool color\n- Chronic fatigue\n- Nausea or vomiting\n- Loss of appetite\n- Tendency to bruise easily ")

# Medical Assistant,       
if (selected=='Medical Assistant'):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    st.title('Chatbot For General Illness Detection')

    training_data = [
        ["I have a sore throat and cough", "Common cold"],
        ["I feel feverish and have body aches", "Influenza"],
        ["I'm experiencing nausea and vomiting", "Stomach flu"],
        ["I have difficulty breathing and chest pain", "Pneumonia"],
        ["I cough up phlegm and have a fever", "Bronchitis"],
        ["I have facial pain and a blocked nose", "Sinusitis"],
        ["I feel a burning sensation while urinating", "Urinary tract infection"],
        ["I have diarrhea and stomach cramps", "Gastroenteritis"],
        ["I have a severe headache and sensitivity to light", "Migraine"],
        ["I sneeze a lot and have itchy eyes", "Allergies"],
    ]

    # Preparing the training data
    symptoms = [data[0] for data in training_data]
    illnesses = [data[1] for data in training_data]

    # Vectorizing symptoms
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(symptoms)

    # Training the classifier
    classifier = MultinomialNB()
    classifier.fit(X, illnesses)

    # Bot's introductory message
    def get_intro_message():
        return "Hello! I'm a medical chatbot. How can I assist you today?"

    # Bot's response to user input
    def get_bot_response(user_input):
        # Vectorize user input
        user_input_vector = vectorizer.transform([user_input])

        # Predict the illness
        predicted_illness = classifier.predict(user_input_vector)
       
         # Check if the user input is valid
        if user_input.strip() == "":
            return "Please enter your symptoms."

        if not any(symptom in user_input for symptom in symptoms):
            return "I'm not sure what you're asking. Please enter a valid symptom."

        else:
            return f"I predict that you may have {predicted_illness[0]} based on the symptoms you provided."

     # Define the UI
    def app():
        st.title("Medical Chatbot")
        st.write(get_intro_message())

        user_input = st.text_input("Enter your symptoms here:")
        if st.button("Predict"):
            if user_input.strip() != "":
                bot_response = get_bot_response(user_input)
                st.markdown("## Diagnosis:")
                st.success(bot_response)
            else:
                st.error("Please enter your symptoms.")

    if __name__ == '__main__':
        app()


# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction System')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction System')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
# breast cancer Prediction Page
if (selected == "Breast Cancer Prediction"):
    
    # page title
    st.title("Breast Cancer Disease Prediction System")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        radius_mean = st.text_input('Radius Mean')
        
    with col2:
        texture_mean= st.text_input('Texture Mean')
        
    with col3:
        perimeter_mean = st.text_input('Perimeter Mean ')
        
    with col4:
        area_mean = st.text_input('Area Mean')
        
    with col5:
        smoothness_mean = st.text_input('Smoothness Mean')
        
    with col1:
        compactness_mean = st.text_input('Compactness Mean')
        
    with col2:
        concavity_mean = st.text_input('Concavity Mean')
        
    with col3:
        concave_points_mean = st.text_input('Concave Point Mean')
        
    with col4:
        symmetry_mean = st.text_input('Symmetry Mean')
        
    with col5:
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean')
        
    with col1:
        radius_se = st.text_input('Radius Se')
        
    with col2:
        texture_se = st.text_input('Texture Se')
        
    with col3:
        perimeter_se = st.text_input('Perimeter Se')
        
    with col4:
        area_se = st.text_input('Area Se')
        
    with col5:
        smoothness_se = st.text_input('Smoothness Se')
        
    with col1:
        compactness_se = st.text_input('Compactness Se')
        
    with col2:
        concavity_se = st.text_input('Concavity Se')

    with col3:
        concave_points_se = st.text_input('Concave points Se')
        
    with col4:
        symmetry_se = st.text_input('Symmetry Se')
        
    with col5:
        fractal_dimension_se = st.text_input('Fractal dimension Se')
        
    with col1:
        radius_worst = st.text_input('Radius Worst')
        
    with col2:
        texture_worst = st.text_input('Texture Worst')
        
    with col3:
        perimeter_worst = st.text_input('Perimeter Worst')
    
    with col4:
        area_worst = st.text_input('Area Worst')

    with col5:
        smoothness_worst = st.text_input('Smoothness Worst')
    
    with col1:
        compactness_worst = st.text_input('Compactness Worst')
    
    with col2:
        concavity_worst = st.text_input('Concavity Worst')
    
    with col3:
        concave_points_worst = st.text_input('Concave Point Worst')
    
    with col4: 
        symmetry_worst = st.text_input('Symmetry Worst')
    
    with col5:
        fractal_dimension_worst = st.text_input('Factal Dimension Worst')

        
    
    
    # code for Prediction
    breast_cancer_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Breast Cancer Test Result"):
        breast_cancer_prediction = breast_cancer_model.predict([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])                          
        
        if (breast_cancer_prediction[0] == 1):
          breast_cancer_diagnosis = "The person has breast cancer disease"
        else:
          breast_cancer_diagnosis = "The person does not have breast cancer disease"
        
    st.success(breast_cancer_diagnosis)


# Kidney Disease Prediction Page
if (selected == "Kidney Disease Prediction"):

    st.title("Kidney Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        age = st.text_input('Age')
    
    with col2:
        blood_pressure = st.text_input('Blood Pressuree')

    with col3:
        specific_gravity = st.text_input('Specific Gravity')
    
    with col4:
        albumin = st.text_input("Albumin")
        
    with col5:
        sugar = st.text_input('Sugar')
    
    with col1:
        red_blood_cells = st.text_input('Red Blood Cells')

    with col2:
        pus_cell = st.text_input('Pus Cell')

    with col3:
        pus_cell_clumps = st.text_input('Pus Cell Clumps')

    with col4:
        bacteria = st.text_input('Bacteria')

    with col5:
        blood_glucose_random = st.text_input('Bloog Glucose Random')

    with col1:
        blood_urea = st.text_input('Blood Urea')

    with col2:
        serum_creatinine = st.text_input('Serum Creatinie')

    with col3:
        sodium = st.text_input('Sodium')

    with col4:
        potassium = st.text_input('Potassium')

    with col5:
        haemoglobin = st.text_input('Haemoglobin')
    
    with col1:
        packed_cell_volume = st.text_input('Packed Cell Volume')

    with col2:
        white_blood_cell_count = st.text_input('White Blood Cell Count')

    with col3:
        red_blood_cell_count = st.text_input('Red Blood Cell Count')

    with col4:
        hypertension = st.text_input('Hypertension')

    with col5:
        diabetes_mellitus  = st.text_input('Diabetes Mellitus')
    
    with col1:
        coronary_artery_disease = st.text_input('Coronary Artery Disease')

    with col2:
        appetite  = st.text_input('Appetite')

    with col3:
        peda_edema = st.text_input('Peda Edema')

    with col4:
        aanemia = st.text_input('Aanemia')

    
    # code for Prediction
    kidney_disease_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Kidney Disease Test Result"):
        kidney_disease_prediction = kidney_disease_model.predict([[age,blood_pressure,specific_gravity,albumin,sugar,red_blood_cells,pus_cell,pus_cell_clumps,bacteria,blood_glucose_random,blood_urea,serum_creatinine,sodium,potassium,haemoglobin,packed_cell_volume,white_blood_cell_count,red_blood_cell_count,hypertension,diabetes_mellitus,coronary_artery_disease,appetite,peda_edema,aanemia]])                          
        
        if (kidney_disease_prediction[0] == 1):
          kidney_disease_diagnosis = "The person has kidney disease"
        else:
          kidney_disease_diagnosis= "The person does not have kideny disease"
        
    st.success(kidney_disease_diagnosis)
    

# Liver Disease Prediction Page
if (selected == 'Liver Disease Prediction'):
    
    # page title
    st.title('Liver Disease Prediction System')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age')
        
    with col2:
        Total_Bilirubin = st.text_input('Total Bilirubin')
        
    with col3:
        Direct_Bilirubin = st.text_input('Direct Bilirubin')
        
    with col1:
        Alkaline_Phosphotase = st.text_input('Alkaline Phosphotase')
        
    with col2:
        Alamine_Aminotransferase = st.text_input('Alamine Aminotransferase')
        
    with col3:
        Aspartate_Aminotransferase = st.text_input('Aspartate Aminotransferase')
        
    with col1:
        Total_Protiens = st.text_input('Total Protiens')
        
    with col2:
        Albumin = st.text_input('Albumin')
        
    with col3:
        Albumin_and_Globulin_Ratio = st.text_input('Albumin and Globulin Ratio')
        
    with col1:
        Gender_Male = st.text_input('Gender Male')
          
     
     
    # code for Prediction
    liver_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Liver Disease Test Result'):
        liver_prediction = liver_disease_model.predict([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender_Male]])                          
        if (liver_prediction[0] == 1):
          liver_diagnosis = 'The person is having liver disease'
          
        else:
          liver_diagnosis = 'The person does not have any liver disease'
          
        
    st.success(liver_diagnosis)
