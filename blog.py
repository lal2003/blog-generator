import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to get response from GPT-2 model
def getBlogResponse(input_text, no_words, blog_style):
    try:
        # Load GPT-2 model
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Generate input prompt
        prompt = f"Write a blog for {blog_style} job profile on the topic: {input_text} within {no_words} words."

        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate response
        output = model.generate(
            inputs.input_ids,
            max_length=150,  # Adjust max_length as needed
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )

        # Decode the output tokens back into text
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Set Streamlit page configuration
st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

# User inputs
input_text = st.text_input("Enter the Blog Topic")

# Create two columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

# Submit button
submit = st.button("Generate")

# Final response
if submit:
    response = getBlogResponse(input_text, no_words, blog_style)
    if response:
        st.write(response)
