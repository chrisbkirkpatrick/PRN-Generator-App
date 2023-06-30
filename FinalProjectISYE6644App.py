import streamlit as st
import timeit
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from itertools import groupby
from numpy import sqrt


st.title("The Random Number Generator App")
st.subheader("What is this app and how do I use it?")
st.markdown("This app is designed to generate Unfiorm(0,1) Pseudo Random Numbers (PRNs) based on different methods, some good and some bad. Choose a generator from the drop down list, enter the number of random variables you wish to generate, and enter any number greater than zero as the initial seed.")
st.subheader("What are the charts displaying?")
st.markdown("The first chart is a simple histogram of the randomly generated numbers. If the generator worked well, you would expect to see a fairly uniform distribution of random numbers with no peaks or gaps.")
st.markdown("The second chart is a 3D scatterplot where the x, y, and z coordinates correspond with the first three random variables generated. The next x, y, and z coordinates would be the next three numbers and so on. If the generator worked well, you should see a fairly random display of points. If the generator didn't work well, you may see distinct patterns or an absence of points to plot if the generator deteriorated quickly.")
st.subheader("What are the metrics displayed?")
st.markdown("There are many different ways to evaluate the utility of a random number generator.")
st.markdown("In particular, we would look for a full cycle length, meaning that the generator was able to generate a sufficient number of random variables before having to repeat itself which would be evident in the chart.")
st.markdown("The Chi Square value measures goodness of fit. To illustrate, imagine that we have 1,000 truly random variables that we divide into 5 buckets. We would expect that there would be 200 values of 0-0.2, 200 values of 0.2-0.4, etc. The Chi Square metrics divides the generated random numbers into these same buckets and then evaluates whether there is a statistically significant difference between the sum of the observed and expected values. A value of alpha is used as a cutoff point where values above alpha cause us to reject the null hypothesis and values below to fail to reject the null hypothesis which is that there is no statistical difference between the observed and the generated values. Chi square values are highlighted red if we reject the null hypothesis and green if we fail to reject the null hypothesis.")
st.markdown("The Up/Down test is a test for determining independence. We would expect that the previously generated random variable should have no effect on the up or down direction of the next variable. This test measures that by analyzing run lengths, or whether subsequent randomly generated numbers are positive or negative when compared with the previous value. This test also relies on an alpha threshold where a red value means we reject the null hypothesis and a green value means we fail to reject the null hypothesis.")
st.markdown("The Autocorrelation test determines whether there is autocorrelation between each subsequent random value and the previous one. This is similar to the Up/Down test and also relies on an alpha threshold level.")
st.subheader("What if I get an error or the numbers look funny?")
st.markdown("This could be due to a coding error, but remember that the generators included are the good, the bad, and the ugly. If any numbers look funny or the generator fails to fully generate the requested amount of random variables, it's very likely that you're working with a bad generator.")

def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|)
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

def chi_square(numbers_to_simulate, rands):
    expected_values = []
    actual_values = []
    x2 = 0

    if numbers_to_simulate < 10:
        k = numbers_to_simulate
    else:
        k = 10

    expected_bins = numbers_to_simulate // k

    for i in range(k):
        expected_values.append(expected_bins)

    increment = 1 / k
    bottom_range = 0

    for j in range(k):
        count = sum(bottom_range <= r < bottom_range + increment for r in rands)
        actual_values.append(count)
        bottom_range += increment

    for l, m in zip(expected_values, actual_values):
        x2 += ((m - l) ** 2) / l

    return x2

def run_test(rands, numbers_to_simulate):
    l = []
    for i, j in enumerate(rands[:-1]):
        if j  < rands[i+1]:
            l.append(1)
        else:
            l.append(0)
    counts = [(i, len(list(g))) for i, g in groupby(l)]
    mu = ((2*numbers_to_simulate)-1)/3
    var = ((16*numbers_to_simulate)-29)/90
    z = (len(counts)-mu)/sqrt(var)
    return abs(z)

def autocorrelation(rands, numbers_to_simulate):
    sum_value = 0
    for i, j in enumerate(rands[:-1]):
        sum_value += j * rands[i+1]
    p = ((12/(numbers_to_simulate - 1))*sum_value)-3
    variance = ((13*numbers_to_simulate)-19)/((numbers_to_simulate - 1)**2)
    z = p/sqrt(variance)
    return abs(z)

#Couldn't get this to work properly so I stuck with the up/down run test
# def above_below(rands, numbers_to_simulate):
#     l = []
#     for i in rands:
#         if i >= 0.5:
#             l.append(1)
#         else :
#             l.append(0)
#     d = []
#     for x, j in enumerate(rands[:-1]):
#         if j  < rands[x+1]:
#             d.append(1)
#         else:
#             d.append(0)
#     counts = [(x, len(list(g))) for i, g in groupby(d)]
#     n1 = sum(l)
#     n2 = len(rands) - sum(l)
#     mu = ((2*n1*n2)/numbers_to_simulate)+0.5
#     var = (2*n1*n2)*((2*n1*n2)-numbers_to_simulate)/((numbers_to_simulate**2)*(numbers_to_simulate-1))
#     z = (len(counts) - ((2*n1*n2)/numbers_to_simulate) - 0.5)/(((2*n1*n2)*((2*n1*n2)-numbers_to_simulate))/(numbers_to_simulate**2)*(numbers_to_simulate - 1))**0.5
#     return abs(z)

def populateMidSquare(seed, numbers_to_simulate):
    count = 0
    rands = []
    while count < numbers_to_simulate :
        if count < 1:
            xi = seed ** 2
            number_str = str(xi)
            middle_four = number_str[(len(number_str)//2) - 2 : (len(number_str)//2) + 2]
            middle_four_digits = int(middle_four)
            ui = middle_four_digits/10000
        else :
            xi = middle_four_digits ** 2
            number_str = str(xi)
            middle_four = number_str[(len(number_str)//2) - 2 : (len(number_str)//2) + 2]
            middle_four_digits = int(middle_four)
            ui = middle_four_digits/10000
        count += 1
        rands.append(ui)
    return rands


#RANDU
def populateRANDU(seed, numbers_to_simulate):
    count = 0
    rands = []
    xi = 0
    previous_xi = 0
    while count < numbers_to_simulate:
        if count < 1:
            xi = (65539 * seed) % (2 ** 31)
            previous_xi = xi
        else:
            xi = (65539 * previous_xi) % (2 ** 31)
            previous_xi = xi
        ui = xi / (2 ** 31)
        count += 1
        rands.append(ui)
    return rands

def populateDesertIsland(seed, numbers_to_simulate):
    count = 0
    rands = []
    xi = 0
    previous_xi = 0
    while count < numbers_to_simulate:
        if count < 1:
            xi = (16807 * seed) % ((2 ** 31)-1)
            previous_xi = xi
        else:
            xi = (16807 * previous_xi) % ((2 ** 31)-1)
            previous_xi = xi
        ui = xi / ((2 ** 31)-1)
        count += 1
        rands.append(ui)
    return rands

def populateTausworthe(seed, numbers_to_simulate):
    seed_bin = bin(seed).replace("0b", "")
    q = len(seed_bin)
    r = 3
    l = []
    for i in seed_bin:
        l.append(int(i))
    for x in range((q * numbers_to_simulate)-q):
        bit = (l[-r] + l[-q]) % 2
        l.append(bit)
    bit_strings = [l[i:i+q] for i in range(0, len(l), q)]
    base10_numbers = [int(''.join(map(str, bits)), 2) for bits in bit_strings]
    rands = [b/(2**q) for b in base10_numbers]
    return rands




# def populateTausworthe(seed, numbers_to_simulate):
#     rands = []
#     count = 0
#
#     while count < numbers_to_simulate:
#         p = 13
#         q = 19
#         r = 32
#         n = 10
#         l = 10
#         decimal = 0
#         bits = ""
#
#         for _ in range(n):
#             x_bit = ((seed >> p) ^ (seed >> q)) & 1
#             decimal = (decimal << 1) | x_bit
#             bits += str(x_bit)
#             seed = ((seed << 1) & ((2 ** r) - 1)) | x_bit
#
#         rand_val = decimal / float((2 ** n) - 1)
#         rand_val_str = bits[-l:]
#         rand_num = int(rand_val_str, 2) / (2 ** l)
#         rands.append(rand_num)
#         count += 1
#
#     return rands

def populateACORN(seed, numbers_to_simulate):
    #arbitrary prime for seed 2
    seed2 = 872621
    count = 0
    rands = []
    xi = 0
    #kth order
    k = 12
    previous_xi_values = [0] * k
    while count < numbers_to_simulate:
        if count < 1:
            xi = (seed2 + seed) % (2 ** 30)
            previous_xi_values[0] = xi
        else:
            if count < k:
                xi_sum = seed2 + seed + sum(previous_xi_values[:count])
            else:
                xi_sum = sum(previous_xi_values)
            xi = xi_sum % (2 ** 30)
            previous_xi_values.pop(0)
            previous_xi_values.append(xi)
        ui = xi / (2 ** 30)
        count += 1
        rands.append(ui)
    return rands


col3, col4, col5, col6 = st.columns(4)
with col3:
    option = st.selectbox(
        'Please select random number generator:',
        ('Mid Square', 'RANDU', 'Tausworthe Generator', 'Desert Island', 'ACORN'))
    st.write('You selected:', option)

with col4:
    numbers_to_simulate = st.number_input('Insert number of random numbers to generate:', value = int(), min_value = 0)
    st.write('You have selected', numbers_to_simulate , 'random numbers')

with col5:
    seed = st.number_input('Insert a seed:', value = int(), min_value = 0)
    st.write('The seed is', seed)

with col6:
    run_code = st.button("Run Code")

if run_code:
    if option == 'Mid Square':
        start = timeit.default_timer()
        rands = populateMidSquare(seed, numbers_to_simulate)
        stop = timeit.default_timer()
        total_time = stop - start
        temp = pd.DataFrame(rands, columns = ["value"])
        fig = px.histogram(temp, x="value")
        st.plotly_chart(fig)
        rands = np.array(rands)
        num_triplets = len(rands) // 3
        x_vals = rands[0:num_triplets*3:3]
        y_vals = rands[1:num_triplets*3:3]
        z_vals = rands[2:num_triplets*3:3]
        min_length = min(len(x_vals), len(y_vals), len(z_vals))
        x_vals = x_vals[:min_length]
        y_vals = y_vals[:min_length]
        z_vals = z_vals[:min_length]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, cmap='Blues')
        ax.view_init(elev=50, azim=65)
        st.pyplot(fig)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Time", round(total_time,5))
        chi_square_value = round(chi_square(numbers_to_simulate, rands),2)
        if chi_square(numbers_to_simulate, rands) > 9.42 :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#FF0000')
        else :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#00B050')
        run_test_value = round(run_test(rands, numbers_to_simulate),2)
        if run_test(rands, numbers_to_simulate) > 1.96:
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#FF0000')
        else :
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#00B050')
        ac_value = round(autocorrelation(rands, numbers_to_simulate),2)
        if autocorrelation(rands, numbers_to_simulate) > 1.96:
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#FF0000')
        else :
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#00B050')
        with st.expander("How were these random numbers calculated?"):
            st.write("The Mid Square Generator is derived using the formula below:")
            st.latex(r'''\text{Start by taking a number seed and squaring it.} \\ Set X_{1} \text{ to the middle four numbers and repeat this process to generate a series of } X_{n} \\
            \\
            X_{0} = 6632 \\
            6632^{2} = 43983424 \\
            X_{1} = 9834 \\
            \text{Take each } X_{n} \text{ and divide by 10000 to obtain a random variable} \\
            \\
            R_{1} = \frac{9834}{10000} = .9834''')
    elif option == 'RANDU':
        start = timeit.default_timer()
        rands = populateRANDU(seed, numbers_to_simulate)
        stop = timeit.default_timer()
        total_time = stop - start
        temp = pd.DataFrame(rands, columns = ["value"])
        fig = px.histogram(temp, x="value")
        st.plotly_chart(fig)
        rands = np.array(rands)
        num_triplets = len(rands) // 3
        x_vals = rands[0:num_triplets*3:3]
        y_vals = rands[1:num_triplets*3:3]
        z_vals = rands[2:num_triplets*3:3]
        min_length = min(len(x_vals), len(y_vals), len(z_vals))
        x_vals = x_vals[:min_length]
        y_vals = y_vals[:min_length]
        z_vals = z_vals[:min_length]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, cmap='Blues')
        ax.view_init(elev=50, azim=65)
        st.pyplot(fig)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Time", round(total_time,5))
        chi_square_value = round(chi_square(numbers_to_simulate, rands),2)
        if chi_square(numbers_to_simulate, rands) > 9.42 :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#FF0000')
        else :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#00B050')
        run_test_value = round(run_test(rands, numbers_to_simulate),2)
        if run_test(rands, numbers_to_simulate) > 1.96:
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#FF0000')
        else :
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#00B050')
        ac_value = round(autocorrelation(rands, numbers_to_simulate),2)
        if autocorrelation(rands, numbers_to_simulate) > 1.96:
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#FF0000')
        else :
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#00B050')
        with st.expander("How were these random numbers calculated?"):
            st.write("The RANDU Generator is calculated using the formula below:")
            st.latex(r'''X_{n+1} = (65539)X_{n}\mod 2^{31}''')
            st.write("The RANDU generator is an infamous case of a Linear Congruential Generator considered to be one of the worst random number generators. The choice of constants is problematic and results in numbers that are not random or uniformally distributed.")
    elif option == 'Tausworthe Generator':
        start = timeit.default_timer()
        rands = populateTausworthe(seed, numbers_to_simulate)
        stop = timeit.default_timer()
        total_time = stop - start
        temp = pd.DataFrame(rands, columns = ["value"])
        fig = px.histogram(temp, x="value")
        st.plotly_chart(fig)
        rands = np.array(rands)
        num_triplets = len(rands) // 3
        x_vals = rands[0:num_triplets*3:3]
        y_vals = rands[1:num_triplets*3:3]
        z_vals = rands[2:num_triplets*3:3]
        min_length = min(len(x_vals), len(y_vals), len(z_vals))
        x_vals = x_vals[:min_length]
        y_vals = y_vals[:min_length]
        z_vals = z_vals[:min_length]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, cmap='Blues')
        ax.view_init(elev=50, azim=65)
        st.pyplot(fig)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Time", round(total_time,5))
        chi_square_value = round(chi_square(numbers_to_simulate, rands),2)
        if chi_square(numbers_to_simulate, rands) > 9.42 :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#FF0000')
        else :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#00B050')
        run_test_value = round(run_test(rands, numbers_to_simulate),2)
        if run_test(rands, numbers_to_simulate) > 1.96:
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#FF0000')
        else :
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#00B050')
        ac_value = round(autocorrelation(rands, numbers_to_simulate),2)
        if autocorrelation(rands, numbers_to_simulate) > 1.96:
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#FF0000')
        else :
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#00B050')
        with st.expander("How were these random numbers calculated?"):
            st.write("The Tausworthe Generator is derived using the formula below:")
            st.write("Start by defining a series of binary digits:")
            #st.latex(r'''\begin{align*} &B_i = \sum_{j=1}^{q} c_j\B_i-j\mod2''')
            st.latex(r'''
            \begin{align*} B_i &= \sum_{j=1}^{q} c_j B_{i-j}\mod2 \\ \text{Convert } B_i \text{ values to Unif(0,1) by: } \\ &\frac{\text{{(l-bits in base 2)}}}{2^l} \\ \text{ex: } 1111_{2}, 1000_{2}, 1101_{2}, 1101_{2} &= \frac{15}{16}, \frac{8}{16}, \frac{13}{16}, \frac{13}{16} \end{align*}''')
    elif option == 'Desert Island':
        start = timeit.default_timer()
        rands = populateDesertIsland(seed, numbers_to_simulate)
        stop = timeit.default_timer()
        total_time = stop - start
        temp = pd.DataFrame(rands, columns = ["value"])
        fig = px.histogram(temp, x="value")
        st.plotly_chart(fig)
        rands = np.array(rands)
        num_triplets = len(rands) // 3
        x_vals = rands[0:num_triplets*3:3]
        y_vals = rands[1:num_triplets*3:3]
        z_vals = rands[2:num_triplets*3:3]
        min_length = min(len(x_vals), len(y_vals), len(z_vals))
        x_vals = x_vals[:min_length]
        y_vals = y_vals[:min_length]
        z_vals = z_vals[:min_length]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, cmap='Blues')
        ax.view_init(elev=50, azim=65)
        st.pyplot(fig)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Time", round(total_time,5))
        chi_square_value = round(chi_square(numbers_to_simulate, rands),2)
        if chi_square(numbers_to_simulate, rands) > 9.42 :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#FF0000')
        else :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#00B050')
        run_test_value = round(run_test(rands, numbers_to_simulate),2)
        if run_test(rands, numbers_to_simulate) > 1.96:
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#FF0000')
        else :
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#00B050')
        ac_value = round(autocorrelation(rands, numbers_to_simulate),2)
        if autocorrelation(rands, numbers_to_simulate) > 1.96:
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#FF0000')
        else :
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#00B050')
        with st.expander("How were these random numbers calculated?"):
            st.write("The Desert Island Generator is calculated using the formula below:")
            st.latex(r'''X_{n+1} = (16807)X_{n}\mod 2^{31} - 1''')
            st.write("The Desert Island Generator is a good form of Linear Congruential Generator, especially when compared with the RANDU Linear Congruential Generator. This generator achieves full cycle lengths and produces numbers that appear to be random and i.i.d.")
    elif option == 'ACORN':
        start = timeit.default_timer()
        rands = populateACORN(seed, numbers_to_simulate)
        stop = timeit.default_timer()
        total_time = stop - start
        temp = pd.DataFrame(rands, columns = ["value"])
        fig = px.histogram(temp, x="value")
        st.plotly_chart(fig)
        rands = np.array(rands)
        num_triplets = len(rands) // 3
        x_vals = rands[0:num_triplets*3:3]
        y_vals = rands[1:num_triplets*3:3]
        z_vals = rands[2:num_triplets*3:3]
        min_length = min(len(x_vals), len(y_vals), len(z_vals))
        x_vals = x_vals[:min_length]
        y_vals = y_vals[:min_length]
        z_vals = z_vals[:min_length]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_vals, y_vals, z_vals, cmap='Blues')
        ax.view_init(elev=50, azim=65)
        st.pyplot(fig)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Time", round(total_time,5))
        chi_square_value = round(chi_square(numbers_to_simulate, rands),2)
        if chi_square(numbers_to_simulate, rands) > 9.42 :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#FF0000')
        else :
            col2.metric("Chi Squared", str(chi_square_value))
            ColourWidgetText(str(chi_square_value), '#00B050')
        run_test_value = round(run_test(rands, numbers_to_simulate),2)
        if run_test(rands, numbers_to_simulate) > 1.96:
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#FF0000')
        else :
            col3.metric("Up/Down Run Test", str(run_test_value))
            ColourWidgetText(str(run_test_value), '#00B050')
        ac_value = round(autocorrelation(rands, numbers_to_simulate),2)
        if autocorrelation(rands, numbers_to_simulate) > 1.96:
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#FF0000')
        else :
            col4.metric("Autocorrelation Test", str(ac_value))
            ColourWidgetText(str(ac_value), '#00B050')
        with st.expander("How were these random numbers calculated?"):
            st.text("The ACORN Generator is calculated using the formula below:")
            st.latex(r'''X_m^n = X_{m-1}^n + X_m^{n-1}\mod  2^{30}''')
            st.text("Where the number of X values added before the modulus can be modified with parameter k.")
else:
    pass
