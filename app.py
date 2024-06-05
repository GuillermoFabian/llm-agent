# importing libraries
import pandas as pd
from typing import List
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, NonNegativeInt
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import gradio as gr

# Environment variables
load_dotenv('.env.local')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4'


# Convert result to DataFrame and save to CSV
def save_to_csv(listings, filename="real_state_listings.csv"):
    # Convert each RealEstateListing object to a dictionary and create a DataFrame
    df = pd.DataFrame([listing.dict() for listing in listings.listings])
    df.to_csv(filename, index=False)



example =  """
Neighborhood: London, UK
Price (USD): 500,000
Bedrooms: 3
Bathrooms: 2
House Size (sqft): 2,000
Description: Welcome to your park oasis in London, United Kindom! This charming house features 3 bedrooms, 2 bathrooms, and breathtaking views of the river.
"""

# Define Pydantic models
class RealEstateListing(BaseModel):
    neighborhood: str = Field(description="The neighborhood where the property is located")
    price: NonNegativeInt = Field(description="The price of the property in the local currency")
    bedrooms: NonNegativeInt = Field(description="The number of bedrooms in the property")
    bathrooms: NonNegativeInt = Field(description="The number of bathrooms in the property")
    house_size: NonNegativeInt = Field(description="The size of the house in square feet")
    description: str = Field(description="A description of the property")
    neighborhood_description: str = Field(description="A description of the neighborhood")

class ListingCollection(BaseModel):
    """
    A collection of real estate listings.
    
    Attributes:
    - listings: List[RealEstateListing]
    """
    listings: List[RealEstateListing] = Field(description="A list of real estate listings")


# Initialize the OutputParser
parser = PydanticOutputParser(pydantic_object=ListingCollection)

prompt = PromptTemplate(
    template="""
        You are an expert Real State Agent.
        Generate {quantity} realistic real estate listings from diverse neighborhoods in {city}. Here's a sample listing:
        {format_instructions}
        Example:\n{example}
        """,
    description="Real State Examples",
    input_variables=["question", "quantity", "city"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def display_similar_docs(house_size, top_3, ammenities, transportation, urban_characteristics):

    query = f"""
    Based on the input data in the context:
    House Size: {house_size}
    Bedrooms: {top_3}
    Top 3 valuable things: {top_3}
    Amenities: {ammenities}
    Transportation: {transportation}
    Urban characteristics: {urban_characteristics}
    Make sure you do not paraphrase the data, and only use the information provided in the available data.
    """

    loader = CSVLoader(file_path='real_state_listings.csv')
    docs = loader.load()

    llm = OpenAI(model_name=MODEL_NAME, temperature=0, max_tokens=2000)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(split_docs, embeddings)
    similar_docs = db.similarity_search(query, k=3)
    

    data = []
    for doc in similar_docs:
        content = doc.page_content.split('\n')
        content_dict = {item.split(': ')[0]: item.split(': ')[1] for item in content}
        metadata_filtered = {key: value for key, value in doc.metadata.items() if key not in ['row', 'source']}
        content_dict.update(metadata_filtered)
        data.append(content_dict)
    
    df = pd.DataFrame(data)
    return df

# Function to generate and display sample data
def generate_sample_data():
    # Assuming example and other necessary variables are defined
    model = ChatOpenAI(model=MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    # Chain the components
    chain = prompt | model | parser
    result = chain.invoke({"example": example, "city": "London", "quantity": 15})
    save_to_csv(result)
    return "Sample data generated and saved successfully."

# Create a Gradio interface
def update_interface(state, button_pressed):
    if button_pressed == "generate":
        message = generate_sample_data()
        return state, message, True  # Enable the 'display' button
    elif button_pressed == "display":
        docs = display_similar_docs()
        return state, docs, True
    return state, "", False  # Initial state, with 'display' button disabled

interface = gr.Interface(
    fn=update_interface,
    inputs=[
        gr.State(),  # To maintain the state across button presses
        gr.Radio(["generate", "display"], label="Select Action")
    ],
    outputs=[
        gr.State(),
        gr.Dataframe(label="Output")
    ],
    live=False,  # Disable live updates, actions only happen when button is pressed
    allow_flagging="never"
)

# Add a button to the loaded interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Real Estate")
    house_size = gr.Textbox(label="How big do you want your house to be?", value="2000 sqft")
    top_3 = gr.Textbox(label="What are 3 most important things for you in choosing this property?", value="Good location, Modern design, High security")
    ammenities = gr.Textbox(label="Which amenities would you like?", value="Pool, Gym, Garage")
    transportation = gr.Textbox(label="Which transportation options are important to you?", value="Close to subway station")
    urban_characteristics = gr.Textbox(label="How urban do you want your neighborhood to be?", value="Very urban")

    btn = gr.Button("Find me the most 3 suitable options")
    output = gr.Dataframe(headers=["Neighborhood", "Price", "Bedrooms", "Bathrooms", "House Size", "Description"])
    btn.click(fn=display_similar_docs, inputs=[house_size, top_3, ammenities, transportation, urban_characteristics], outputs=output)


#demo.launch(generate_sample_data())
demo.launch()

