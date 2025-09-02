import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Set up Streamlit Page Configuration ---
st.set_page_config(
    page_title="Disease Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

# --- Load Data and Model ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('health_lifestyle_classification.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file 'health_lifestyle_classification.csv' not found. Please ensure the file is in the same directory.")
        return None

@st.cache_resource
def load_model_and_preprocessors():
    """Load model and all preprocessing components"""
    try:
        model = joblib.load('rforest_model.pkl')
        scaler = joblib.load('scaler.pkl')  # You'll need to save this during training
        selector = joblib.load('selector.pkl')  # You'll need to save this during training
        feature_names = joblib.load('model_features.pkl')
        return model, scaler, selector, feature_names
    except FileNotFoundError as e:
        
        return None, None, None, None

# Alternative function if you haven't saved the preprocessors yet
@st.cache_resource
def load_model_only():
    try:
        model = joblib.load('rforest_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'rforest_model.pkl' not found.")
        return None

# Load data and model
df = load_data()
model, scaler, selector, model_features = load_model_and_preprocessors()

# Fallback if preprocessors aren't available
if model is None:
    model = load_model_only()

# --- Sidebar Navigation ---
st.sidebar.title('🏥 Disease Risk Prediction')
page = st.sidebar.radio(
    'Navigate to:',
    ['📊 Data Exploration', '📈 Model Performance', '🔮 Make Prediction']
)

# --- Page 1: Data Exploration ---
if page == '📊 Data Exploration':
    st.header('Disease Risk Dataset - Exploratory Data Analysis 🔍')
    st.markdown('***')
    
    if df is not None:
        # Dataset overview
        st.subheader('📦 Dataset Overview')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(df.columns)-1}")
        with col3:
            healthy_count = len(df[df['target'] == 'healthy']) if 'target' in df.columns else 0
            st.metric("Healthy", f"{healthy_count:,}")
        with col4:
            diseased_count = len(df[df['target'] == 'diseased']) if 'target' in df.columns else 0
            st.metric("Diseased", f"{diseased_count:,}")

        st.markdown('---')

        # Target distribution
        if 'target' in df.columns:
            st.subheader('🎯 Target Distribution')
            target_counts = df['target'].value_counts()
            fig_target = px.pie(values=target_counts.values, names=target_counts.index, 
                              title='Distribution of Disease Risk (Target Variable)')
            st.plotly_chart(fig_target)

        st.markdown('---')

        # Feature distributions
        st.subheader('📊 Feature Distributions')
        
        numerical_features = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'heart_rate', 
                            'glucose', 'insulin', 'calorie_intake', 'sugar_intake', 
                            'screen_time', 'stress_level', 'mental_health_score', 'training_hours']
        
        available_numerical = [col for col in numerical_features if col in df.columns]
        
        if available_numerical:
            selected_feature = st.selectbox(
                'Select a numerical feature to visualize:',
                available_numerical,
                index=0
            )

            col1, col2 = st.columns(2)
            with col1:
                if 'target' in df.columns:
                    fig_hist = px.histogram(df, x=selected_feature, color='target', 
                                          title=f'Distribution of {selected_feature.replace("_", " ").title()}',
                                          template='plotly_white')
                else:
                    fig_hist = px.histogram(df, x=selected_feature, 
                                          title=f'Distribution of {selected_feature.replace("_", " ").title()}',
                                          template='plotly_white')
                st.plotly_chart(fig_hist)

            with col2:
                if 'target' in df.columns:
                    fig_box = px.box(df, x='target', y=selected_feature, 
                                   title=f'{selected_feature.replace("_", " ").title()} by Disease Status',
                                   template='plotly_white')
                else:
                    fig_box = px.box(df, y=selected_feature, 
                                   title=f'Distribution of {selected_feature.replace("_", " ").title()}',
                                   template='plotly_white')
                st.plotly_chart(fig_box)

        st.markdown('---')

        # Feature relationships
        st.subheader('🔗 Feature Relationships')
        if len(available_numerical) >= 2:
            col3, col4 = st.columns(2)
            with col3:
                x_axis = st.selectbox('X-axis', available_numerical, index=0)
            with col4:
                y_axis = st.selectbox('Y-axis', available_numerical, 
                                    index=1 if len(available_numerical) > 1 else 0)

            if 'target' in df.columns:
                fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color='target', 
                                       hover_data=['age', 'bmi'] if 'age' in df.columns and 'bmi' in df.columns else None,
                                       title=f'{x_axis.replace("_", " ").title()} vs {y_axis.replace("_", " ").title()}',
                                       template='plotly_white')
            else:
                fig_scatter = px.scatter(df, x=x_axis, y=y_axis,
                                       title=f'{x_axis.replace("_", " ").title()} vs {y_axis.replace("_", " ").title()}',
                                       template='plotly_white')
            st.plotly_chart(fig_scatter)

        # Correlation heatmap
        if len(available_numerical) > 2:
            st.subheader('🔥 Correlation Heatmap')
            corr_matrix = df[available_numerical].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', 
                               aspect='auto', title='Feature Correlation Matrix')
            st.plotly_chart(fig_corr)

# --- Page 2: Model Performance ---
elif page == '📈 Model Performance':
    st.header('Model Evaluation Dashboard 📈')
    st.markdown('***')

    model_accuracies = {
        'SVM': 0.7009,
        'Random Forest': 0.7006,
        'K-Nearest Neighbors': 0.6355
    }

    st.subheader('🏆 Model Accuracy Comparison')
    fig_acc = px.bar(
        x=list(model_accuracies.keys()),
        y=list(model_accuracies.values()),
        labels={'x': 'Model', 'y': 'Accuracy'},
        title='Accuracy of Different Models',
        color=list(model_accuracies.values()),
        color_continuous_scale='viridis'
    )
    fig_acc.update_yaxes(range=[0.6, 0.75])
    st.plotly_chart(fig_acc)

    st.markdown('---')

    st.subheader('📊 Confusion Matrix')
    st.markdown("Performance visualization of the Random Forest model on test data.")
    st.info("💡 To show the actual confusion matrix, load your real y_test and y_pred from model training.")
    
    cm_data = [[4200, 800], [900, 2100]]
    cm_labels = ['Healthy', 'Diseased']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Disease Prediction - Confusion Matrix (Example)')
    st.pyplot(plt)

# --- Page 3: Make a Prediction ---
elif page == '🔮 Make Prediction':
    st.header('Disease Risk Prediction 🔮')
    st.markdown('***')

    if model is None:
        st.error("❌ Cannot make predictions - model not loaded.")
        st.stop()

    # Check if preprocessors are available
    has_preprocessors = scaler is not None and selector is not None
    if not has_preprocessors:
        st.warning("⚠️ Preprocessors (scaler, selector) not found. Predictions may be inaccurate.")
        st.info("To get accurate predictions, save your scaler and selector during model training.")

    st.markdown("🏥 **Enter your health and lifestyle information to predict disease risk**")

    # Create a list of the features that require one-hot encoding
    categorical_cols = ['gender', 'marital_status', 'occupation', 'family_history',
                        'insurance', 'sleep_quality', 'healthcare_access', 'diet_type',
                        'exercise_type', 'caffeine_intake', 'sunlight_exposure',
                        'pet_owner', 'mental_health_support', 'device_usage']
    
    with st.form("prediction_form"):
        st.markdown("### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider('Age', 18, 100, 35)
            gender = st.selectbox('Gender', ['Male', 'Female'])
            marital_status = st.selectbox('Marital Status',
                                        ['Single', 'Married', 'Divorced', 'Widowed'])
        
        with col2:
            height = st.slider('Height (cm)', 140, 210, 170)
            weight = st.slider('Weight (kg)', 40, 140, 70)
            bmi = weight / ((height/100) ** 2)
            st.write(f"**Calculated BMI: {bmi:.1f}**")
            
        with col3:
            occupation = st.selectbox('Occupation',
                                    ['Software Engineer', 'Doctor', 'Teacher', 'Student',
                                     'Manager', 'Nurse', 'Lawyer', 'Engineer', 'Other'])
            family_history = st.selectbox('Family History of Disease', ['Yes', 'No'])
            insurance = st.selectbox('Health Insurance', ['Yes', 'No'])

        st.markdown("### 🩺 Health Metrics")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            blood_pressure = st.slider('Blood Pressure (systolic)', 80, 200, 120)
            heart_rate = st.slider('Heart Rate (bpm)', 50, 150, 75)
            cholesterol = st.slider('Cholesterol (mg/dL)', 150, 350, 200)
            
        with col5:
            glucose = st.slider('Blood Glucose', 70, 200, 100)
            insulin = st.slider('Insulin Level', 5, 50, 15)
            stress_level = st.slider('Stress Level (0-10)', 0, 10, 5)
            
        with col6:
            mental_health_score = st.slider('Mental Health Score (0-10)', 0, 10, 7)
            sleep_quality = st.selectbox('Sleep Quality', ['Poor', 'Fair', 'Good', 'Excellent'])
            healthcare_access = st.selectbox('Healthcare Access', ['Easy', 'Moderate', 'Difficult'])

        st.markdown("### 🍎 Lifestyle Factors")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            diet_type = st.selectbox('Diet Type',
                                   ['Omnivore', 'Vegetarian', 'Vegan', 'Keto', 'Paleo'])
            calorie_intake = st.slider('Daily Calories', 1200, 4000, 2000)
            sugar_intake = st.slider('Daily Sugar (grams)', 0, 150, 50)
            
        with col8:
            exercise_type = st.selectbox('Primary Exercise',
                                       ['None', 'Cardio', 'Strength', 'Mixed'])
            training_hours = st.slider('Weekly Exercise Hours', 0, 20, 3)
            caffeine_intake = st.selectbox('Caffeine Intake',
                                         ['None', 'Low', 'Moderate', 'High'])
            
        with col9:
            screen_time = st.slider('Daily Screen Time (hours)', 0, 16, 6)
            sunlight_exposure = st.selectbox('Sunlight Exposure', ['Low', 'Medium', 'High'])
            pet_owner = st.selectbox('Pet Owner', ['Yes', 'No'])
            meals_per_day = st.slider('Meals per Day', 1, 6, 3)

        st.markdown("### 🧠 Mental Health & Support")
        col10, col11 = st.columns(2)
        with col10:
            mental_health_support = st.selectbox('Mental Health Support', ['Yes', 'No'])
            device_usage = st.selectbox('Device Usage Level', ['Low', 'Medium', 'High'])
        
        submitted = st.form_submit_button("🔮 Predict Disease Risk", use_container_width=True)
    
    if submitted:
        try:
            # Prepare user input data
            user_input_data = {
                'age': age, 'height': height, 'weight': weight, 'bmi': bmi, 'blood_pressure': blood_pressure,
                'heart_rate': heart_rate, 'cholesterol': cholesterol, 'glucose': glucose, 'insulin': insulin,
                'stress_level': stress_level, 'mental_health_score': mental_health_score, 'calorie_intake': calorie_intake,
                'sugar_intake': sugar_intake, 'training_hours': training_hours, 'screen_time': screen_time,
                'meals_per_day': meals_per_day, 'gender': gender, 'marital_status': marital_status,
                'occupation': occupation, 'family_history': family_history, 'insurance': insurance,
                'sleep_quality': sleep_quality, 'healthcare_access': healthcare_access,
                'diet_type': diet_type, 'exercise_type': exercise_type, 'caffeine_intake': caffeine_intake,
                'sunlight_exposure': sunlight_exposure, 'pet_owner': pet_owner,
                'mental_health_support': mental_health_support, 'device_usage': device_usage
            }
            
            # Create DataFrame and encode categorical variables
            user_df = pd.DataFrame([user_input_data])
            user_df_encoded = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)
            
            # If we have the full preprocessing pipeline
            if has_preprocessors and model_features is not None:
                # Align features with training data
                user_df_aligned = user_df_encoded.reindex(columns=model_features, fill_value=0)
                
                # Apply scaling to numerical features
                numerical_features = user_df_aligned.select_dtypes(include=['int64', 'float64']).columns
                user_df_scaled = user_df_aligned.copy()
                user_df_scaled[numerical_features] = scaler.transform(user_df_scaled[numerical_features])
                
                # Apply feature selection
                user_df_final = selector.transform(user_df_scaled)
                
            else:
                # Fallback: try to use the data as-is (less accurate)
                st.warning("Using fallback preprocessing - results may be less accurate")
                
                if model_features is not None:
                    user_df_final = user_df_encoded.reindex(columns=model_features, fill_value=0)
                else:
                    # Last resort: use all encoded features
                    user_df_final = user_df_encoded
            
            # Make predictions
            prediction = model.predict(user_df_final)
            prediction_proba = model.predict_proba(user_df_final)
            
            # Debug information (remove in production)
            with st.expander("🔍 Debug Information"):
                st.write(f"Input shape: {user_df_final.shape}")
                st.write(f"Prediction: {prediction}")
                st.write(f"Probabilities: {prediction_proba[0] if prediction_proba is not None else 'None'}")
                if hasattr(model, 'classes_'):
                    st.write(f"Model classes: {model.classes_}")
            
            # Handle prediction results
            # Your model maps healthy=0, diseased=1
            if prediction[0] == 0:
                prediction_label = "healthy"
                risk_level = "Low"
            else:
                prediction_label = "diseased"
                risk_level = "High"
            
            # Extract probabilities (model trained with 0=healthy, 1=diseased)
            if prediction_proba is not None:
                prob_healthy = prediction_proba[0][0]  # Probability of class 0 (healthy)
                prob_diseased = prediction_proba[0][1]  # Probability of class 1 (diseased)
            else:
                prob_healthy = 1.0 if prediction_label == "healthy" else 0.0
                prob_diseased = 1.0 if prediction_label == "diseased" else 0.0
            
            # Display results
            if prediction_label == "healthy":
                st.success(f"### 🟢 The model predicts **LOW RISK** (Healthy)")
            else:
                st.warning(f"### 🔴 The model predicts **HIGH RISK** (Disease Risk)")
            
            st.markdown("### 📊 Risk Assessment")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("🟢 Healthy Probability", f"{prob_healthy:.1%}")
            with col_res2:
                st.metric("🔴 Disease Risk Probability", f"{prob_diseased:.1%}")
            
            # Create probability visualization
            proba_df = pd.DataFrame({
                'Status': ['Healthy', 'Diseased'],
                'Probability': [prob_healthy, prob_diseased]
            })
            fig_proba = px.bar(proba_df, x='Status', y='Probability',
                             title='Risk Assessment Probabilities',
                             color='Probability',
                             color_continuous_scale='RdYlGn_r',
                             text='Probability')
            fig_proba.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_proba.update_layout(showlegend=False, yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig_proba, use_container_width=True)
            
            # Risk interpretation
            st.markdown("### 💡 Risk Interpretation")
            if prediction_label == 'healthy':
                st.success(f"✅ **Low Risk ({prob_healthy:.1%})**: Based on your data, you have a lower risk of disease. Keep up the great work!")
            else:
                st.warning(f"⚠️ **Higher Risk ({prob_diseased:.1%})**: The model suggests an elevated disease risk. Consider consulting a healthcare professional and reviewing your lifestyle factors.")
            
            # Health recommendations
            st.markdown("### 🎯 Health Recommendations")
            recommendations = []
            if bmi > 25:
                recommendations.append("🏃‍♂️ Consider weight management through a balanced diet and regular exercise.")
            if bmi < 18.5:
                recommendations.append("🍽️ Consider healthy weight gain through proper nutrition.")
            if stress_level > 7:
                recommendations.append("🧘‍♀️ Practice stress management techniques like meditation or yoga.")
            if training_hours < 2:
                recommendations.append("💪 Increase physical activity to at least 150 minutes per week.")
            if screen_time > 8:
                recommendations.append("📱 Reduce screen time and take regular breaks.")
            if sleep_quality in ['Poor', 'Fair']:
                recommendations.append("😴 Focus on improving sleep quality and duration.")
            if mental_health_score < 5:
                recommendations.append("🧠 Consider seeking mental health support or counseling.")
            if cholesterol > 240:
                recommendations.append("❤️ Monitor cholesterol levels and consider dietary changes.")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("🎉 You're maintaining good health habits! Keep it up!")
            
            # Show confidence level
            confidence = max(prob_healthy, prob_diseased)
            if confidence > 0.8:
                confidence_text = "High"
                confidence_color = "🟢"
            elif confidence > 0.6:
                confidence_text = "Medium" 
                confidence_color = "🟡"
            else:
                confidence_text = "Low"
                confidence_color = "🔴"
            
            st.markdown("---")
            st.markdown(f"**Model Confidence**: {confidence_color} {confidence_text} ({confidence:.1%})")
            st.caption("⚠️ This prediction is for educational purposes only and should not replace professional medical advice.")
            
            if prediction_label == "healthy" and prob_healthy > 0.7:
                st.balloons()
            
        except Exception as e:
            st.error(f"❌ **An unexpected error occurred during prediction**: {str(e)}")
            st.info("Please verify that your model file and preprocessing files are correctly saved and loaded.")
            
            # Show more detailed error information for debugging
            import traceback
            with st.expander("🔍 Detailed Error Information"):
                st.code(traceback.format_exc())