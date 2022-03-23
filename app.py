
import streamlit as st
import pandas as pd


def main():
    page = st.sidebar.selectbox("Select a Page",["Project", "raw data","analysed data"])

    #First Page
    if page == "Project":
        homepage()

    #Second Page
    elif page == "raw data":
        rawdata()
    
    #Third Page
    elif page == "analysed data":
        analyseddata()
        
def homepage():
    
    st.title('Project')
    st.write('xxxx')
 
def rawdata():
    
    st.title('Raw data')
    st.write('xx')
    
    DATA_URL = ('Data/petiteconversation.csv')
    df = pd.read_csv(DATA_URL)
    st.write(df) 
  
 
    
def analyseddata():
    
    st.title('Analysed data')
    st.write('xxx')
 

if __name__ == "__main__":
    main()
    
    
    
