import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from statsmodels.stats.weightstats import ztest as ztest
from tqdm.notebook import tqdm
from scipy.stats import mannwhitneyu
from scipy.stats import anderson_ksamp
from numpy import sqrt, abs, round
from scipy.stats import norm
from PIL import Image

st.set_page_config(layout="wide")

st.title('Statistical Significance Test Calculator')

tabs = st.tabs(["Home", "Z-Test","T-Test"])
tab_home = tabs[0]
tab_ztest = tabs[1]
tab_ttest = tabs[2]
url = "https://en.wikipedia.org/wiki/Kinetic_theory_of_gases"

t_test_critical = []

def load_data_ttable():
    return pd.DataFrame(
        {
            "df": [120,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,60,120],
            "0.2": [1.282,3.078,1.886,1.638,1.533,1.476,1.44,1.415,1.397,1.383,1.372,1.363,1.356,1.35,1.345,1.341,1.337,1.333,1.33,1.328,1.325,1.323,1.321,1.319,1.318,1.316,1.315,1.314,1.313,1.311,1.31,1.296,1.289],
            "0.1": [1.645,6.314,2.92,2.353,2.132,2.015,1.943,1.895,1.86,1.833,1.812,1.796,1.782,1.771,1.761,1.753,1.746,1.74,1.734,1.729,1.725,1.721,1.717,1.714,1.711,1.708,1.706,1.703,1.701,1.699,1.697,1.671,1.658],
            "0.05": [1.96,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.16,2.145,2.131,2.12,2.11,2.101,2.093,2.086,2.08,2.074,2.069,2.064,2.06,2.056,2.052,2.048,2.045,2.042,2,1.98],
            "0.02": [2.326,31.821,6.965,4.541,3.747,3.365,3.143,2.998,2.896,2.821,2.764,2.718,2.681,2.65,2.624,2.602,2.583,2.567,2.552,2.539,2.528,2.518,2.508,2.5,2.492,2.485,2.479,2.473,2.467,2.462,2.457,2.39,2.358],
            "0.01": [2.576,63.656,9.925,5.841,4.604,4.032,3.707,3.499,3.355,3.25,3.169,3.106,3.055,3.012,2.977,2.947,2.921,2.898,2.878,2.861,2.845,2.831,2.819,2.807,2.797,2.787,2.779,2.771,2.763,2.756,2.75,2.66,2.617],
            "0.001": [3.291,636.578,31.6,12.924,8.61,6.869,5.959,5.408,5.041,4.781,4.587,4.437,4.318,4.221,4.14,4.073,4.015,3.965,3.922,3.883,3.85,3.819,3.792,3.768,3.745,3.725,3.707,3.689,3.674,3.66,3.646,3.46,3.373],
        }
    )

with tab_home:
    st.markdown('''Statistical significance calculation is a widely used method in scientific research to determine the probability that the results of an experiment 
or study are due to chance or some other non-causal factor. In this article, we will discuss what statistical significance is, how it is calculated, 
and the factors that influence it.''')

    st.header('What is Statistical Significance?')
    st.markdown('''Statistical significance is a measure of how likely it is that an observed result is not due to chance. In other words, it is a way of assessing the 
strength of the evidence in favor of a particular hypothesis. To determine statistical significance, researchers use statistical tests to calculate the probability 
that the observed result is due to chance.

Statistical tests compare the observed data to what would be expected under a null hypothesis. The null hypothesis is the idea that there is no difference or relationship 
between two variables or groups. If the observed data is significantly different from what would be expected under the null hypothesis, then we can reject the null hypothesis 
and conclude that there is a significant relationship between the variables or groups.
''')

    st.header('Calculating Statistical Significance')
    st.markdown('''The most common method for calculating statistical significance is the p-value. The p-value is the probability of obtaining a result as extreme or more extreme 
than the observed result, assuming the null hypothesis is true. If the p-value is less than a predetermined significance level (usually 0.05 or 0.01), then we reject the 
null hypothesis and conclude that the result is statistically significant.

For example, suppose we conduct a study to compare the mean heights of two groups of people. The null hypothesis is that there is no difference in mean height between the two 
groups. We collect data and calculate the mean height and standard deviation for each group. We then use a t-test to calculate the p-value. If the p-value is less than 0.05, 
we reject the null hypothesis and conclude that there is a statistically significant difference in mean height between the two groups.
''')

    st.header('Factors Affecting Statistical Significance')
    st.markdown('''Several factors can affect the statistical significance of a result, including sample size, effect size, variability, and significance level.

**:blue[Sample Size]**: The larger the sample size, the more likely it is that small differences between groups or variables will be detected, leading to greater statistical significance.

**:blue[Effect Size]**: The effect size is a measure of the magnitude of the difference between groups or variables. The larger the effect size, the more likely it is that the result will be statistically significant.

**:blue[Variability]**: The variability of the data can also affect statistical significance. If the data is highly variable, then it may be more difficult to detect a difference or relationship between variables.

**:blue[Significance Level]**: The significance level is the probability threshold for rejecting the null hypothesis. The most commonly used significance levels are 0.05 and 0.01. A lower significance level (e.g., 0.01) requires stronger evidence to reject the null hypothesis, leading to fewer statistically significant results.
''')

    st.header('Common Types of Significance Testing:')
    st.markdown("- Z-Test: Used for continous metrics")
    st.markdown("- T-Test: Used for continous metrics")
    st.markdown("- Chi-square Test: Used for conversion metrics")
    st.markdown("- Proportion Test: Used for conversion metrics")




with tab_ztest:
    def twoSampZ(X1, X2, sd1, sd2, n1, n2):
        mudiff = 0
        pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
        z = ((X1 - X2) - mudiff)/pooledSE
        pval = 2*(norm.sf(abs(z)))
        return round(z, 3), round(pval, 4)

    st.header("Z-Test")
    st.markdown('''The z-test is a statistical test used to compare a sample mean to a population mean, given that the population standard deviation is known. It is named after the standard 
normal distribution, which is also known as the Z-distribution. The z-test is used when the sample size is large enough that the sampling distribution of the mean can be approximated by a 
normal distribution. The test determines whether the sample mean is significantly different from the population mean. It is often used in quality control, manufacturing, and medical research 
to make decisions about product quality, process control, or treatment effectiveness.

The z-test is based on the following formula:

z = (x - μ) / (σ / √n)''')
    image = Image.open('distribution.png')
    st.image(image, caption='Perform Z-test', width = 400)

    st.header("Z-Test Calculator for two independent samples")
    test_mean = st.number_input('Insert mean of test group')
    control_mean = st.number_input('Insert mean of control group')
    test_stddev = st.number_input('Population standard deviation of test group')
    control_stddev = st.number_input('Population standard deviation of control group')
    test_sample = st.number_input('Sample size of test group')
    control_sample = st.number_input('Sample size of control group')




    # z_statistic,p_value = twoSampZ(test_mean,control_mean,test_stddev,control_stddev,test_sample,control_sample)
    button_click = st.button('Calculate')

    if button_click:
        z_statistic,p_value = twoSampZ(test_mean,control_mean,test_stddev,control_stddev,test_sample,control_sample)
        st.write('Mean of test group = %5.1f, Mean of control group = %5.1f' % (test_mean, control_mean))
        st.write('A/B test evaluation, lift = %.2f%%' % ((test_mean/control_mean-1)*100))
        st.write('A/B test evaluation, z_statistic = %.2f' % (z_statistic))
        st.write('A/B test evaluation, p-value = %.2f' % (p_value))
        if p_value < 0.05:
            st.write('''Mean difference between test and control group is statistical significant for p value 0.05. Hence, it is safe to conclude that difference between test and control group should
                be due to some changes''')
        else:
            st.write('''Mean difference between test and control group is not statistical significant for p value 0.05. Hence, we cannot conclude the test and group are different''')
    else:
        st.write('No Calculation')

    st.header("Z-Test for two independent samples")
    st.markdown('''The z-test for two independent samples is a statistical test used to compare the means of two independent populations. It is used when the samples are independent of each other, 
        and the populations are assumed to be normally distributed with known variances.

The z-test for two independent samples is based on the following formula:

z = (x1 - x2) / √(σ1^2 / n1 + σ2^2 / n2)

Where:

x1 is the sample mean of the first sample
x2 is the sample mean of the second sample
σ1 is the population standard deviation of the first sample
σ2 is the population standard deviation of the second sample
n1 is the sample size of the first sample
n2 is the sample size of the second sample
The z-score obtained from this formula represents the number of standard deviations that the difference between the two sample means is from zero. A z-score greater than the critical value 
obtained from a z-table or z-calculator indicates that the difference between the two means is statistically significant.

To use the z-test for two independent samples, the following steps can be followed:

- State the null and alternative hypotheses. The null hypothesis is that the means of the two populations are equal, while the alternative hypothesis is that they are not equal.
- Collect the two independent samples, and calculate the sample means and standard deviations.
- Calculate the z-score using the formula above.
- Determine the critical value using a z-table or z-calculator for the chosen level of significance.
- Compare the calculated z-score to the critical value. If the calculated z-score is greater than the critical value, then the null hypothesis can be rejected in favor of the alternative hypothesis.

For example, suppose a researcher wants to test whether there is a significant difference in the mean height of men and women. The researcher collects two independent samples of 50 men and 50 women 
and finds that the mean height for men is 175 cm with a standard deviation of 5 cm, while the mean height for women is 162 cm with a standard deviation of 4 cm. The null hypothesis is that the mean 
height of men and women is equal, while the alternative hypothesis is that they are not equal. Using a significance level of 0.05, the critical value for a two-tailed test is ±1.96.

The z-score can be calculated as follows:
z = (175 - 162) / √(5^2 / 50 + 4^2 / 50) = 12 / 1.08 = 11.11

Since the calculated z-score (11.11) is much greater than the critical value (1.96), the null hypothesis can be rejected in favor of the alternative hypothesis. The researcher can conclude that 
there is a statistically significant difference in the mean height of men and women. In summary, the z-test for two independent samples is a useful statistical tool for comparing the means of 
two independent populations. It assumes that the populations are normally distributed with known variances and can be used to test whether the difference between the sample means is statistically 
significant.''')


with tab_ttest:
    def twoSampT(X1, X2, sd1, sd2, n1, n2):
        mudiff = 0
        pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
        t = ((X1 - X2) - mudiff)/pooledSE
        pval = 2*(norm.sf(abs(t)))
        return round(t, 3), round(pval, 4)

    st.header("T-Test")
    st.markdown('''A t-test for two independent samples is used to compare the means of two independent groups to determine if they are significantly different from each other. This test is appropriate 
when the data are continuous and normally distributed, and the samples are independent of each other. Here are the steps to perform a t-test for two independent samples:''')
    image = Image.open('distribution.png')
    st.image(image, caption='Perform T-test', width = 400)

    st.header("T-Test Calculator for two independent samples")
    test_mean_ttest = st.number_input('Insert mean of test group', key = "1")
    control_mean_ttest = st.number_input('Insert mean of control group', key = "2")
    test_stddev_ttest = st.number_input('Sample standard deviation of test group', key = "3")
    control_stddev_ttest = st.number_input('Sample standard deviation of control group', key = "4")
    test_sample_ttest = st.number_input('Sample size of test group', key = "5")
    control_sample_ttest = st.number_input('Sample size of control group', key = "6")




    # z_statistic,p_value = twoSampZ(test_mean,control_mean,test_stddev,control_stddev,test_sample,control_sample)
    button_click = st.button('Calculate', key = "7")

    if button_click:
        t_statistic,p_value = twoSampZ(test_mean_ttest,control_mean_ttest,test_stddev_ttest,control_stddev_ttest,test_sample_ttest,control_sample_ttest)
        st.write('Mean of test group = %5.1f, Mean of control group = %5.1f' % (test_mean_ttest, control_mean_ttest))
        st.write('A/B test evaluation, lift = %.2f%%' % ((test_mean_ttest/control_mean_ttest-1)*100))
        st.write('A/B test evaluation, t_statistic = %.2f' % (t_statistic))
        st.write('A/B test evaluation, degrees of freedom = %.2f' % (test_sample_ttest+control_sample_ttest-2))
        st.write('''Once we obtained the critical value, you need to compare the t-statistic using the below table for the corresponding degrees of freedom. If t-statistic is greater than critical value,
            then differnce between test and control group is statistically significant, otherwise not''')
    else:
        st.write('No Calculation')

    st.header("T-statistic table")
    df_tscore = load_data_ttable()

    # Display the dataframe 
    # Boolean to resize the dataframe, stored as a session state variable
    # st.checkbox("Use container width", value=True, key="use_container_width")
    st.dataframe(df_tscore, use_container_width=True)

    st.header("T-Test for two independent samples")
    st.markdown('''Define the null and alternative hypotheses:
The null hypothesis (H0) is that there is no significant difference between the means of the two groups. The alternative hypothesis (Ha) is that there is a significant difference between the means 
of the two groups.

**[Determine the level of significance]**:
Choose a level of significance (α) to determine the cutoff value for rejecting the null hypothesis. The most commonly used level of significance is α=0.05, which corresponds to a 95% confidence level.

**[Calculate the t-statistic]**:
The t-statistic is calculated as:

t = (x1 - x2) / √(s1^2/n1 + s2^2/n2)

where x1 and x2 are the means of the two samples, s1 and s2 are the standard deviations of the two samples, and n1 and n2 are the sample sizes of the two samples.

**[Determine the degrees of freedom]**:
The degrees of freedom (df) for the t-test for two independent samples is calculated as:

df = (n1 + n2 - 2)

**[Find the critical value]**:
The critical value is the value of the t-statistic that corresponds to the level of significance and the degrees of freedom. This value can be found in a t-table or calculated using software.

**[Compare the t-statistic to the critical value]**:
If the absolute value of the calculated t-statistic is greater than the critical value, then the null hypothesis can be rejected, and it can be concluded that the means of the two groups are significantly different. If the absolute value of the calculated t-statistic is less than the critical value, then the null hypothesis cannot be rejected, and it can be concluded that there is not enough evidence to support a significant difference between the means of the two groups.

In summary, a t-test for two independent samples is a statistical test used to determine if the means of two independent groups are significantly different from each other. The test involves calculating the t-statistic and comparing it to a critical value based on the level of significance and the degrees of freedom.''')
    



    







