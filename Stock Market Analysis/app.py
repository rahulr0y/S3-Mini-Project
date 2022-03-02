import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))


def predict(input):
    # input = np.array([[dif2, dif1, hl1, hl2]]).astype(np.float)
    prediction = model.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def result(input):
    outcome = model.predict(input)
    return outcome


def calc(tue_open, tue_close, tue_high, tue_low, wed_close, wed_high, wed_low):
    dif2 = (tue_open - tue_close)/tue_open * 100
    dif1 = (tue_open - wed_close)/tue_open * 100
    hl2 = (tue_high - tue_low)/tue_close * 100
    hl1 = (wed_high - wed_low)/wed_close * 100
    input = np.array([[dif2, dif1, hl1, hl2]]).astype(np.float)
    return input

def main():
    st.title('Stock Market Analysis')
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">BANK NIFTY OPTIONS </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # tue_open = st.number_input()
    tue_open = st.number_input("Tuesday :- Open")
    tue_high = st.number_input("Tuesday :- High")
    tue_low = st.number_input("Tuesday :- Low")
    tue_close = st.number_input("Tuesday :- Close")
    wed_high = st.number_input("Wednesday :- High")
    wed_low = st.number_input("Wednesday :- Low")
    wed_close = st.number_input("Wednesday :- Close")

    # hl2 = st.text_input("HL_2","Type Here")

    profit_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> There is a chance for profit. Continue with the strategy!</h2>
       </div>
    """
    loss_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> There is chance of loss. It is better to exit now to minimize the loss!</h2>
       </div>
    """

    if st.button("Predict"):
        x_values = calc(tue_open, tue_close, tue_high, tue_low, wed_close, wed_high, wed_low)
        output=predict(x_values)
        outcome = result(x_values)
        st.success('The probability of percent change more than 1% is {}'.format(output))

        if outcome == 1:
            st.markdown(loss_html,unsafe_allow_html=True)
        else:
            st.markdown(profit_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()