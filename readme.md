# GRACE Follow-On微波测距系统与光照

# 1. 概述

由于太阳光照会影响卫星测距系统、微波天线与模数转化系统温度，其成为低低卫卫跟踪重力卫星星间距测量中高频噪声的一个可能来源。研究太阳光照对星间距测量的影响有助于后期我国重力卫星任务中温控仪器设备设计与制造，提高我国重力卫星星间距测量精度，进而提升重力场反演精度。可以将GFO双星K/Ka波段带偏星间距表示为以下形式：
$$
\begin{equation}
	R_K(t)=\rho_K(t)+\delta \rho_K^{solar\_illumination} + \delta \rho_K ^{other}
\end{equation}
$$

$$
\begin{equation}
	R_{Ka}(t)=\rho_{Ka}(t)+\delta \rho_{Ka}^{solar\_illumination} + \delta \rho_{Ka} ^{other}
\end{equation}
$$

因此，考虑以下两种信号来判断星载微波测距系统与光照相关性：

1. 电离层改正
2. 带偏星间距

其中，带偏星间距主频较低，主频振幅较大，因此，高通滤波后的高频噪声可能不明显；电离层改正则与带偏星间距线性相关，且主频振幅较小，有利于窥视高频噪声特性。

# 2. 日心-地心坐标系

为判断卫星是否接受太阳直射，建立以下日心-地心坐标系，由记号$\odot$表示。

![image-20210108125004506](C:\Users\LHS10\AppData\Roaming\Typora\typora-user-images\image-20210108125004506.png)
$$
\begin{equation}
	\bar{\mathbf{r}}_{\odot}^{sat}=\mathbf{R}_2(-\delta_{\odot})\mathbf{R}_3(\alpha_{\odot}) \cdot \bar{\mathbf{r}}_{i}^{sat}
\end{equation}
$$
上式中，$\bar{\mathbf{r}}_{\odot}^{sat}$表示卫星在日心-地心坐标系下的位置矢量，$\bar{\mathbf{r}}_{i}^{sat}$表示卫星在惯性系下的位置矢量，$\alpha_{\odot}$表示某时刻以地球为中心时太阳赤经，$\delta_{\odot}$表示某时刻以地球为中心时太阳赤纬==（该两项参数由刘伟杰提供）==。上式所表示的物理含义为将某时刻惯性系下卫星位置矢量绕惯性系z轴逆时针旋转太阳赤经角度，而后又在新坐标系下相对新y轴顺时针旋转赤纬角度，即得日心-地心坐标系下卫星位置矢量。

上图中构建了地球阴影模型，由此可以确定，当$\bar{x}_{\odot}^{sat}>0$时，卫星一定接受太阳直射；当$\bar{x}_{\odot}^{sat}<0$时，若满足
$$
\begin{equation}
	y_{\odot, sat}^2+z_{\odot,sat}^2 \le r^2_E
\end{equation}
$$
卫星不接受太阳直射。

# 3. 处理方法

![analysis_flow](E:\lhsPrograms\Projects\kbr_a2b_oop\report\analysis_flow.png)

一般而言，根据GRACE卫星微波测距系统误差分析文献，高通滤波的截止频率为0.02Hz。该频率应用于电离层改正时效果较好，但应用于瞬时带偏星间距时，效果较差，因此，瞬时带偏星间距的高通截止频率为0.1Hz。

# 4. 处理结果

经过上述处理流程，可以得到以下结果：

## 4.1 电离层改正与光照关系

![post_analysis_iono_residual_0.02hz](E:\lhsPrograms\Projects\kbr_a2b_oop\images\post_analysis_iono_residual_0.02hz.png)

## 4.2 K波段带偏带偏星间距与光照

![dowr_k_solar](E:\lhsPrograms\Projects\kbr_a2b_oop\images\dowr_k_solar.png)

对比光照与否高频噪声的功率与方差，其中处于光照时
$$
\begin{equation}
	power_{in} = 8.12721151365314e-07 \\
	variance_{in} = 2.031940019448061e-11
\end{equation}
$$

不处于光照时，
$$
\begin{equation}

	power_{not\_in} = 5.297189793053613e-07\\
	variance_{not\_in} = 4.424941157733108e-13
\end{equation}
$$
比较后发现，处于光照时的双星K波段星间距高频噪声功率为不处于光照时的1.6倍，而噪声方差为不处于光照时的50倍。因此，光照将会显著增强微波测距系统的高频噪声，进而影响重力场信号反演精度。