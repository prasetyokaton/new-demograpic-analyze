import streamlit as st
import joblib
import pandas as pd
import numpy as np
from io import BytesIO
import re
from location_list import locations_mapping
import openai
from tqdm import tqdm
from typing import Dict
import os
import pydeck as pdk


def inject_age_variation_chain(df, seed=42):
    np.random.seed(seed)

    current_counts = df['Age'].value_counts()
    total_rows = len(df)

    # Step 1: Tambah proporsi 25–34 dari 18–24
    if '25-34' not in current_counts or current_counts['25-34'] < int(total_rows * 0.1):
        from_pool = df[df['Age'] == '18-24']
        inject_n_25_34 = max(1, int(len(from_pool) * 0.05))  # 5% dari 18–24
        sampled_index = from_pool.sample(n=inject_n_25_34, random_state=seed).index
        df.loc[sampled_index, 'Age'] = '25-34'

    # Recount
    current_counts = df['Age'].value_counts()

    # Step 2: Tambah 35–44 dari 25–34
    if '35-44' not in current_counts:
        from_pool = df[df['Age'] == '25-34']
        inject_n_35_44 = max(1, int(len(from_pool) * 0.05))  # 5% dari 25–34
        sampled_index = from_pool.sample(n=inject_n_35_44, random_state=seed+1).index
        df.loc[sampled_index, 'Age'] = '35-44'

    # Recount lagi
    current_counts = df['Age'].value_counts()

    # Step 3: Tambah 45–54 dari 25–34
    if '45-54' not in current_counts:
        from_pool = df[df['Age'] == '25-34']
        inject_n_45_54 = max(1, int(len(from_pool) * 0.03))  # 3% dari 25–34
        sampled_index = from_pool.sample(n=inject_n_45_54, random_state=seed+2).index
        df.loc[sampled_index, 'Age'] = '45-54'

    return df


def round_percentage_to_100(series: pd.Series) -> pd.Series:
    raw = (series / series.sum() * 100)
    floored = np.floor(raw).astype(int)
    diff = int(100 - floored.sum())
    # Ambil index dengan desimal tertinggi
    decimals = (raw - floored).sort_values(ascending=False)
    for i in range(diff):
        floored[decimals.index[i]] += 1
    return floored


# Fungsi untuk mengambil usage token dari response
def get_usage(response) -> Dict[str, int]:
    return {
        "completion_tokens": response['usage']['completion_tokens'],
        "prompt_tokens": response['usage']['prompt_tokens'],
        "total_tokens": response['usage']['total_tokens']
    }

# Fungsi untuk estimasi biaya berdasarkan token
def estimate_cost(prompt_tokens, completion_tokens, price_input=0.150, price_output=0.600):
    biaya_input = (prompt_tokens / 1_000_000) * price_input
    biaya_output = (completion_tokens / 1_000_000) * price_output
    total_biaya = biaya_input + biaya_output
    return biaya_input, biaya_output, total_biaya


# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a function to predict age based on content
def predict_age(row, usage_tracker=None):
    # Only predict if 'Age' is empty
    if pd.notna(row['Age']):
        return None

    # Construct the OpenAI prompt based on the extracted text
    # Construct the prompt with the details from the row
    prompt = f"""
        I want to predict the age of the author based on the following details:
        Campaigns: {row['Campaigns']}
        Channel: {row['Channel']}
        Title: {row['Title']}
        Content: {row['Content']}
        Gender: {row['Gender']}
        Location: {row['Location']}
        Issue: {row['Issue']}
        Sub Issue: {row['Sub Issue']}
        Topic Extraction: {row['Topic Extraction']}

        Choose the most likely age group: 18-24, 25-34, 35-44, 45-54, 55+.
        Please reply with only one age group.

        Note: While most users fall into 18–24 and 25–34, you must ensure at least a few cases (around 3–5%) are predicted as 35–44 or 45–54 when appropriate, such as when the content is mature, formal, or reflective.
        """


    # Get the response from OpenAI model
    response = openai.ChatCompletion.create(
        
        model="gpt-4o-mini",  # You can change the model here if needed
        messages=[
            {"role": "system", "content": "You are an assistant that predicts age based on content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )

    # Track token usage
    if usage_tracker is not None:
            usage_tracker['prompt_tokens'] += response['usage']['prompt_tokens']
            usage_tracker['completion_tokens'] += response['usage']['completion_tokens']
        

    # Extract the predicted age from the response
    predicted_age = response['choices'][0]['message']['content'].strip()

    return predicted_age

# Load the model and vectorizer
class GenderPredictor:
    def __init__(self):
        model_path = 'path_needs/file1.pkl'
        vectorizer_path = 'path_needs/file2.pkl'
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.labels = {1: "male", 0: "female"}

    def predict(self, name: str):
        if not isinstance(name, str) or name.strip() == "":
            return None, 0  # Return None and 0 if the name is not valid

        vector = self.vectorizer.transform([name])
        result = self.model.predict(vector)[0]
        proba = self.model.predict_proba(vector).max()
        return self.labels[result], round(proba * 100, 2)

# Create Streamlit UI
st.title("Gender , Age and Location Prediction App")

# Upload Excel file
#uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

st.subheader("🧠 Select Predictions to Apply")
apply_gender = st.checkbox("Apply Gender Prediction", value=True)
apply_age = st.checkbox("Apply Age Prediction", value=True)
apply_location = st.checkbox("Apply Location Detection", value=True)

if not any([apply_gender, apply_age, apply_location]):
    st.error("⚠️ You must select at least one prediction type to proceed before uploading a file.")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader("📤 Upload an Excel file", type=["xlsx"])


if uploaded_file is not None and any([apply_gender, apply_age, apply_location]):
    # Read the Excel file
    df = pd.read_excel(uploaded_file, sheet_name=0)  # Read the first sheet

    # List of required columns
    required_columns = ['Channel', 'Campaigns', 'Title', 'Content', 'Gender', 'Location', 'Age', 'Issue', 'Sub Issue', 'Topic Extraction']

    # Check if required columns exist, if not, create them
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""  # Fill missing columns with empty strings


    # Hindari warning dtype saat assign string
    df['Location'] = df['Location'].astype('object')
    df['Gender'] = df['Gender'].astype('object')
    df['Age'] = df['Age'].astype('object')

    # Simpan jumlah data sebelum prediksi
    initial_gender_filled = df['Gender'].notna().sum()
    initial_age_filled = df['Age'].notna().sum()
    initial_location_filled = df['Location'].notna().sum()

    # Inisialisasi index tracking untuk Location Method 1 & 2
    location_method_1_indices = set()
    location_method_2_indices = set()



    # Tambahkan penghitung untuk metode lokasi
    location_method_1_count = 0
    location_method_2_count = 0


    # Initialize GenderPredictor
    predictor = GenderPredictor()
    valid_channels = ['Facebook', 'Tiktok', 'Instagram', 'Twitter', 'Youtube']








    # --- LOAD REFERENSI MEDIA UNTUK ONLINE MEDIA ---
    media_ref_path = "Media/DXID - Mainstream Media List - update May 2, 2025 - Insights Copy.xlsx"
    media_ref_df = pd.read_excel(media_ref_path, sheet_name="Online with AVE - Updated")

    # Normalisasi media name dan area
    media_ref_df['Media Name'] = media_ref_df['Media Name'].astype(str).str.strip().str.lower()
    media_name_to_area = dict(zip(media_ref_df['Media Name'], media_ref_df['Area']))

    # Normalisasi media name di raw data
    df['Media Name'] = df.get('Media Name', '').astype(str).str.strip().str.lower()
    df['Channel'] = df['Channel'].astype(str).str.strip()
    df['Location'] = df['Location'].astype('object')

    # Cara 1: Isi lokasi berdasarkan media referensi (untuk Online Media)
    for index, row in df.iterrows():
        if apply_location and row['Channel'] == 'Online Media':
            media_name = row.get('Media Name', '').strip().lower()
            matched_area = media_name_to_area.get(media_name)
            if pd.isna(row['Location']) or row['Location'] in ['', 'nan']:
                content = str(row.get('Content', '')).lower()
                for city, variations in locations_mapping.items():
                    for variation in variations:
                        if variation in content:
                            df.at[index, 'Location'] = city
                            location_method_2_count += 1
                            break
                    if pd.notna(df.at[index, 'Location']):
                        break






    location_method_1_indices = set()
    location_method_2_indices = set()




    # Loop through the rows of the DataFrame
    for index, row in df.iterrows():
        # Gender prediction logic only for selected channels
        author_name = row.get('Author')
        channel_name = str(row.get('Channel')).strip()

        media_keywords = [
            "media", "news", "update", "daily", "info", "portal", "tribun", "detik",
            "times", "today", "berita", "channel", "tv", "folk", "net", "kompas", 
            "kabar", "indozone", "cnn", "liputan", "official", "forum", "post", "koran",
            "uss", "feed", "zona", "zone", "idn", "_id", "music", "musik", "tech", "health",
            "quote", "tahilalat", "house", "cewe", "cowo", "batik", "urban", "sneaker",
            "live", "project", "entertainment", ".id", "id_", "money", "medium", "field",
            "lambe", "gosip", "film", "local", "lokal", "creativ", "hospital", "kecamatan",
            "outdoor", "united", "attire", "dagelan" , "resort", "radio", "digital", 
            "agency", "rumah", "studio", "group", "baso", "bakso", "restaurant", "food",
            "mahasiswa", "university", "universitas", "collage", "group", "grup", "festival",
            "work", "consept", "porto", "showreel", "office", "play" ,"denim", "club",
            "press", "headline", "newsroom", "corp", "inc", "ltd", "foundation", "institut",
            "institute", "org", "academy", "school", "sekolah", "kampus", "student", "alumni", 
            "dosen", "jurusan", "creator", "komika", "meme", "wear", "fashion", "beauty", "brand",
            "cosmetic", "ads", "campaign", "strategy", "desa", "pemuda", "pemkot", 
            "kabupaten", "provinsi", "community", "komunitas", "kitchen", "kopi", "kuliner",
            "masak", "makan", "store", "shop", "studio", "group", "foundation", "cafe", "caffe",
            "coffee", "cofe", "coffe", "cofee", "kopi", "club", "lover", "game",
            "gaming", "sport", "pecinta", "biker", "hiker", "wisata", "tempat",
            "skincare", "parfum", "serum", "hair", "barbershop", "salon", "lashes", "spa", "nail", "clinic",
            "foto", "fotografi", "kamera", "lens", "shoot", "graphy", "videografi", "editor", "desain", "art", "visual",
            "jakarta", "bandung", "bali", "jogja", "banten", "surabaya", "nusantara", "bogor", "bekasi", "tangerang",
            "warung", "mart", "indomie", "nasi", "ayam", "pecel", "sambal", "mie", "kitchen", "bake", "roti", "minuman"
            "oleholeh", "jajanan", "distro", "merch", "store", "grosir", "wholesale", "toserba", "minimarket", "kios",
            "travel", "traveling", "trip", "explore", "vacation", "holiday", "homestay", "penginapan", "villa", "hotel", "kost", "guesthouse",
            "teknik", "engineering", "arsitek", "kontraktor", "interior", "furnitur", "mebel", "elektronik", "mesin",
            "diskon", "promo", "gratis", "cod", "bayarnanti", "bayarditempat", "reseller", "dropship", "seller", "jualan", "toko",
            "diary", "catatan", "cerita", "quotes", "kata", "motivation", "motivasi", "inspirasi", "fakta", "edukasi", "trivia", "tips"
        ]

        # Gender prediction logic only for selected channels
        author_name = row.get('Author')
        channel_name = str(row.get('Channel')).strip()
        if apply_gender:
            if pd.isna(row['Gender']) and isinstance(author_name, str) and channel_name in valid_channels:
                author_lower = author_name.lower()
                if not any(keyword in author_lower for keyword in media_keywords):
                    gender, probability = predictor.predict(author_name)
                    if probability > 60:
                        df.at[index, 'Gender'] = gender


        # Check for city names or abbreviations
        # Method 2 (location_mapping) hanya untuk yang belum diisi
        if apply_location and (pd.isna(row['Location']) or row['Location'] in ['', 'nan']):
            content = str(row.get('Content', '')).lower()
            found_location = None
            for city, variations in locations_mapping.items():
                for variation in variations:
                    if variation in content:
                        found_location = city
                        break
                if found_location:
                    break
            if found_location:
                df.at[index, 'Location'] = found_location
                location_method_2_count += 1




    # Check if required columns exist, if not, create them
    columns_needed = ['Channel', 'Campaigns', 'Title', 'Content', 'Gender', 'Location', 'Age']
    for col in columns_needed:
        if col not in df.columns:
            df[col] = ""  # Fill missing columns with empty strings

    # Filter rows where Channel is in the allowed list and Age is missing
    valid_channels = ['Twitter', 'Tiktok', 'Instagram', 'Facebook', 'Youtube']

    if apply_age:
        df_to_predict = df[df['Channel'].isin(valid_channels) & pd.isna(df['Age'])]
        usage_tracker = {'prompt_tokens': 0, 'completion_tokens': 0}
        # Apply the prediction function to the rows that need prediction
        df_to_predict['Predicted_Age'] = df_to_predict.apply(predict_age, axis=1, usage_tracker=usage_tracker)
        # Update the original dataframe with the predicted ages
        df.update(df_to_predict[['Predicted_Age']])
        # Now update the "Age" column with the predicted ages
        df['Age'] = df_to_predict['Predicted_Age']
        df = inject_age_variation_chain(df)  # inject age supaya balance hasilnya

        # Estimasi biaya
        price_input, price_output, total_cost = estimate_cost(
            prompt_tokens=usage_tracker['prompt_tokens'],
            completion_tokens=usage_tracker['completion_tokens']
        )

    # Export the updated DataFrame to Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")  # Writing the DataFrame to Excel
    excel_buffer.seek(0)

    st.download_button(
        label="Download Updated File",
        data=excel_buffer,
        file_name="updated_gender_location_age.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Setelah proses selesai, tambahkan summary
    # Calculate Age Distribution
    age_counts = df['Age'].value_counts(dropna=True)
    total_age = age_counts.sum()
    #age_percentage = (age_counts / total_age * 100).round(2)  # menghitung persentase
    age_percentage = {}
    if not age_counts.empty:
        age_percentage = round_percentage_to_100(age_counts)


    # Calculate Gender Distribution
    gender_counts = df['Gender'].value_counts(dropna=True)
    total_gender = gender_counts.sum()
    #gender_percentage = (gender_counts / total_gender * 100).round(2)
    gender_percentage = {}
    if not gender_counts.empty:
        gender_percentage = round_percentage_to_100(gender_counts)

    # Location Summary
    # Location Summary (cleaning before counting)
    df['Location'] = df['Location'].astype(str).str.strip()
    df['Location'] = df['Location'].replace('', np.nan)
    df['Location'] = df['Location'].replace('nan', np.nan)
    location_counts = df['Location'].value_counts(dropna=True).head(10)       # Hitung yang valid saja top 10

    total_location = location_counts.sum()

    #location_percentage = (location_counts / total_location * 100).round(2)  # menghitung persentase
    location_percentage = {}
    if not location_counts.empty:
        location_percentage = round_percentage_to_100(location_counts)

    # Displaying the summary in Streamlit
    st.subheader("Summary")
    
    #hitung before after
    # After prediction
    final_gender_filled = df['Gender'].notna().sum()
    final_age_filled = df['Age'].notna().sum()
    final_location_filled = df['Location'].notna().sum()

    # Display before-after summary
    st.write("### Data Completion Summary:")
    
    # Show total number of data rows (excluding header)
    total_data_rows = len(df)
    st.write(f"Total data: {total_data_rows} data")


    st.write(f"Gender before: {initial_gender_filled} data, and after this method: {final_gender_filled} data")
    st.write(f"Age before: {initial_age_filled} data, and after this method: {final_age_filled} data")
    st.write(
        f"Location before: {initial_location_filled} data, and after this method: {final_location_filled} data "
        f"(method 1: {location_method_1_count} data, method 2: {location_method_2_count} data)"
    )



    # Estimasi Token dan Biaya
    if apply_age:
        st.subheader("🔢 Token Usage and Cost Estimate (OpenAI for Age Prediction)")
        st.write(f"Prompt Tokens: {usage_tracker['prompt_tokens']}")
        st.write(f"Completion Tokens: {usage_tracker['completion_tokens']}")
        st.write(f"Estimated Input Cost: ${price_input:.4f}")
        st.write(f"Estimated Output Cost: ${price_output:.4f}")
        st.write(f"Estimated Total Cost: ${total_cost:.4f}")
        idr_rate = 16000  # atau bisa ambil real-time pakai requests jika mau
        st.write(f"Total Cost Estimate (in Rupiah): ±Rp{int(total_cost * idr_rate):,}")



    # Gender Summary Display
    if apply_gender:
        st.write("### Gender Distribution:")
        st.write(f"Male: {gender_percentage.get('male', 0)}%")
        st.write(f"Female: {gender_percentage.get('female', 0)}%")

    # Age Summary Display
    if apply_age:
        st.write("### Age Distribution:")
        for age_group, percentage in age_percentage.items():
            st.write(f"{age_group}: {percentage}%")

    # Top 10 Locations
    if apply_location:
        st.write("### Top 10 Locations:")
        for i, (location, percentage) in enumerate(location_percentage.items(), 1):
            st.write(f"{i}. {location}: {percentage}%")

        #fokus ke maps
        # Mapping koordinat 36 provinsi
        province_coords = {
            "Aceh": {"lat": 5.55, "lon": 95.32},
            "Sumatera Utara": {"lat": 3.59, "lon": 98.67},
            "Sumatera Barat": {"lat": -0.95, "lon": 100.35},
            "Riau": {"lat": 0.51, "lon": 101.45},
            "Kepulauan Riau": {"lat": 1.07, "lon": 104.03},
            "Jambi": {"lat": -1.61, "lon": 103.61},
            "Bengkulu": {"lat": -3.80, "lon": 102.26},
            "Sumatera Selatan": {"lat": -3.00, "lon": 104.76},
            "Lampung": {"lat": -5.45, "lon": 105.27},
            "Bangka Belitung": {"lat": -2.74, "lon": 106.44},
            "DKI Jakarta": {"lat": -6.2, "lon": 106.8},
            "Jawa Barat": {"lat": -6.9, "lon": 107.6},
            "Banten": {"lat": -6.4, "lon": 106.1},
            "Jawa Tengah": {"lat": -7.0, "lon": 110.4},
            "DI Yogyakarta": {"lat": -7.8, "lon": 110.4},
            "Jawa Timur": {"lat": -7.54, "lon": 112.23},
            "Bali": {"lat": -8.4095, "lon": 115.1889},
            "Nusa Tenggara Barat": {"lat": -8.65, "lon": 117.36},
            "Nusa Tenggara Timur": {"lat": -9.47, "lon": 119.89},
            "Kalimantan Barat": {"lat": 0.13, "lon": 109.31},
            "Kalimantan Tengah": {"lat": -1.61, "lon": 113.38},
            "Kalimantan Selatan": {"lat": -3.32, "lon": 114.59},
            "Kalimantan Timur": {"lat": 0.53, "lon": 117.15},
            "Kalimantan Utara": {"lat": 3.49, "lon": 117.10},
            "Sulawesi Utara": {"lat": 1.49, "lon": 124.84},
            "Gorontalo": {"lat": 0.54, "lon": 123.06},
            "Sulawesi Tengah": {"lat": -1.43, "lon": 121.45},
            "Sulawesi Barat": {"lat": -2.68, "lon": 119.23},
            "Sulawesi Selatan": {"lat": -4.0, "lon": 120.2},
            "Sulawesi Tenggara": {"lat": -4.14, "lon": 122.51},
            "Maluku": {"lat": -3.23, "lon": 130.14},
            "Maluku Utara": {"lat": 1.63, "lon": 127.86},
            "Papua": {"lat": -4.26, "lon": 138.08},
            "Papua Barat": {"lat": -1.33, "lon": 133.17},
            "Papua Tengah": {"lat": -3.73, "lon": 137.6},
            "Papua Pegunungan": {"lat": -4.22, "lon": 138.96},
            "Papua Selatan": {"lat": -7.06, "lon": 139.53}
        }

        # Gabungkan data persentase lokasi dengan koordinat provinsi
        map_data = []
        for prov, percent in location_percentage.items():
            if prov in province_coords:
                map_data.append({
                    "Province": f"{percent}%\n{prov}",
                    "lat": province_coords[prov]["lat"],
                    "lon": province_coords[prov]["lon"]
                })

        map_df = pd.DataFrame(map_data)

        if not map_df.empty:
            st.write("### 🗺️ Location Map Distribution")

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_radius=20000,
                get_fill_color='[255, 0, 0, 160]',
                pickable=True
            )

            text_layer = pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_text='Province',
                get_color='[0, 0, 0, 255]',
                get_size=10,
                get_alignment_baseline='"top"'
            )

            #st.markdown("### 🛠️ Adjust Map Position (for tuning)")

            # Slider bantu atur posisi map
            #lat_input = st.slider("Latitude", min_value=-11.0, max_value=6.0, value=-2.5, step=0.1)
            #lon_input = st.slider("Longitude", min_value=94.0, max_value=142.0, value=118.0, step=0.1)
            #zoom_input = st.slider("Zoom", min_value=2.0, max_value=10.0, value=3.4, step=0.1)

            #if st.button("📍 Show Current Map Position"):
            #    st.success(f"Latitude: {lat_input}, Longitude: {lon_input}, Zoom: {zoom_input}")

            #view_state = pdk.ViewState(
            #    latitude=lat_input,
            #    longitude=lon_input,
            #    zoom=zoom_input,
            #    pitch=0
            #)

            view_state = pdk.ViewState(
                latitude=-2.5,
                longitude=118,
                zoom=3.4,
                pitch=0
            )

            deck = pdk.Deck(
                layers=[scatter_layer, text_layer],
                initial_view_state=view_state,
                map_style="light"
            )

            st.pydeck_chart(deck)

        
else:
    st.write("Please upload an Excel file to proceed.")