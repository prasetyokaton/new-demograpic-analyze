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
        Please reply in this exact format:

        Age Group: [your chosen age group]
        Confidence: [confidence level in percent, from 0 to 100]
        Interest: [one-word or short phrase describing main interest if found, otherwise write 'unknown']

        Note: While most users fall into 18–24 and 25–34, ensure at least 3–5% are predicted as 35–44 or 45–54 if appropriate (e.g. content is mature, reflective, or formal).
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
    result_text = response['choices'][0]['message']['content'].strip()

    # Extract using regex
    match = re.search(r"Age Group:\s*(\d{2}-\d{2}|\d{2}\+)\s*Confidence:\s*(\d+)", result_text)
    if match:
        age_group = match.group(1)
        confidence = int(match.group(2))
        if confidence > 80:
            return age_group  # return only if confidence threshold met
        else:
            return None
    else:
        return None  # fallback if format is unexpected