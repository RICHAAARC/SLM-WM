"""保存跨平台位精确的标准正态 float32 分位数表.

表中第 ``i`` 个值定义为
``round_binary32(Phi^-1((i + 0.5) / 1048576))``.运行时只解码冻结的
IEEE-754 位模式,不调用平台 ``libm``.模块仅嵌入正半轴位模式的递增差分,
负半轴通过精确设置符号位重建;压缩表示不参与科学身份,完整表原始大端字节
SHA-256 才是权威身份.
"""

from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
import struct
import zlib


NORMAL_QUANTILE_INDEX_BITS = 20
NORMAL_QUANTILE_COUNT = 1 << NORMAL_QUANTILE_INDEX_BITS
NORMAL_QUANTILE_TABLE_VERSION = (
    "standard_normal_midpoint_icdf_float32_table20_v1"
)
NORMAL_QUANTILE_TABLE_SHA256 = (
    "70abf440a7f3670147965ffa52f5aaa639dab97f6282b68f3a9a1b1ce5e6cf5a"
)
NORMAL_QUANTILE_REFERENCE_VERIFICATION_PROTOCOL = (
    "mpfr_192bit_erf_midpoint_bracket_and_newton_v2"
)
NORMAL_QUANTILE_REFERENCE_PRECISION_BITS = 192
NORMAL_QUANTILE_REFERENCE_NEWTON_ITERATIONS = 3
NORMAL_QUANTILE_REFERENCE_MPFR_ROUNDING_MODE = (
    "round_to_nearest_ties_to_even"
)
NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST = (
    "e270eb3d2dd29d52b98a1e501065ed0df8debfb71129881e766f98224d36f227"
)
NORMAL_QUANTILE_MAXIMUM_CDF_CELL_WIDTH = 1.0 / NORMAL_QUANTILE_COUNT
NORMAL_QUANTILE_IDEAL_MIDPOINT_KS_DISTANCE = 0.5 / NORMAL_QUANTILE_COUNT
NORMAL_QUANTILE_FLOAT32_CDF_ROUNDING_ERROR_BOUND = (
    1.4386451474557305e-08
)
NORMAL_QUANTILE_FLOAT32_KS_DISTANCE_BOUND = (
    NORMAL_QUANTILE_IDEAL_MIDPOINT_KS_DISTANCE
    + NORMAL_QUANTILE_FLOAT32_CDF_ROUNDING_ERROR_BOUND
)
_POSITIVE_QUANTILE_COUNT = NORMAL_QUANTILE_COUNT // 2
_POSITIVE_DELTA_VARINT_SHA256 = (
    "f2b818db8d931acfe7e8a131f7381e638601d0b4653ba34b6b98c1b9c3ea2506"
)
_POSITIVE_DELTA_VARINT_ZLIB_BASE85 = (
    "c-ri}>DzYcRn~b_Z2M^W;urpaHe)3usd=8O5{68bs?=N|2?-$~kOaaUNMcf%h-5Gj5Q1T-0qqt<11dDy!77;tZQE`@Y-"
    "DUyu$8uLY!y27zR!K{Ygog1u4`R;@8?b3d+)W*^LMUwJ#X+h{8oE^=dZr=2mj13{K2RG+h6<H-+0nb|Imk@^vK`&-"
    "t!)R{`Gsl<GgSE_)EX-"
    "<1hWz;{K!GzWC=pc>d3R@cgDW_MiXiXaCN7Kl^u!+pu=@u3!1{pM2o6{`Q%>5s5ai^%Eca?|${2fA)C)TA`n?qW#Vv`EUN<Q&0NC"
    "Pd({uJ&M)rDg6^09&(BfKZ=h$c;2V~|F2KqfA@P^Awomsj-4~}r*Sdygc(PzeE8;n?g`<FKY8oNU;6m<-"
    "7o$4OItNVueG_(dGr1qr-9Dd!t)s?jqPM@pSx<k&GD)2vo-"
    "d0lC%A5*MI2z51#*_^UG?hjAY$2)|lMS$(Y~Msq?4ucjwZrH|Aw)PHa23#m=l#w*MQat?vEoHTLpn_nXg8#;5newwM^3-"
    "@lu)vCgJVk)4nH_wW1c_kH$zKl^>fRlCe|=2^Mhb5EP^_i*BVCvtXXof0M6pFMjjcXK<o*7oS+o@vhX+*5eI?M8O8m#sZ+&fcD%="
    "S-"
    "|SlRK3^H)oQYWvBW1$i~`R<GZ8Vd1qvwH1EW9&)?%;{>=P$pVsUBPyDNQ@mJr!)AQcH;r{zq+j{@>jrUKP&7XY#%2PQr;}c`~yzL"
    "pAd)V4?<NZ6C9;ItdBJ3~y`FDTT1D~}#wUY@txAs}vk!{Y{SH9kK-tG)nlYRa>_q8)NHD`N0K65j7V*c#t*4&exQ19OSgsrF7-"
    "`*-&V=`~^boez-8JWu8-4D&(*e2W81UEN-I=(ZyIrpqR{=L&sPv2i#dwbT#np4~5TPABy?}N?S%AXjU-"
    "^0nW+t=B1XSO_V?UUxGGp5#<o_oqZ|61Qq<6qpz#@vZHyX&2jy}bt6YI<(QPOUdT>x_M=Ezj9va{ktKbp46=bk64f^cvgiOwCMUa"
    "~WItXE1JybM}&sP4C~&*oaK5cM3-"
    "4?!~9}#nzk5pj%Am@kkOmYcE^#HobIyzSaablRJga&6(t8BlD4s*v`ma#&*sLbIzK5>OTJ3XU4zuG+gg_;@|n{>lgpryZ-Dxt_ME"
    "j{tXYD+Q0C?ZqIvQC#O8H+4cjgZ$7Zwod>4Q&zqhzn?JeDT;9~aX4aUPf5HQsZO=;QY_6MSZ|yh9+t?SMdCI&q=Fs)_$0oNs@qyi"
    "*7M<FDZ%eY)DH*5E_;<`d;dJ}cGR>s3PTGF+G`j817@wPY0+y_^KWl2v_PVF$>%E?yJw0zCcW)~?XKTHD?&R3c+9#f7wU^UQ<5XT"
    "Y^Q8R!Ey#M?r#o49f}dP-Bmac$|H8KYG@QQu^m<eIY2MD3bnYfUGrF<fX=83bCvP{Ax3w+XW_sOZA9?=N+}Y^HoHJ&gH@mVMr|Ey"
    "){!W~~v;N%diG5^Q8*9<+r*gLT<JZ_c9mzH`(djuSoSSN!{k>1@gKs^x-"
    "FYK!tCQBxGdAX&A(Q=V&e+bO+s)g_zV@>AGyckqDQ=RVj%~!Za<=E5w#{F7;>mbU(_h$Pd-g{5WY*ODi8*QRiFtn~XMP6F-"
    "p<_Ij^uCT<|EVVoP@b;&Rb(@J-"
    "+rtY<6^OpMPhs=jTtazrDs8d7Jz2Gq=VmvzzsA$lX|H8=aUl7n#o4oO{+2wXWNrG;3qcsd&C+GCsWzw#CG}?XBE)8`+zAG=C=}<)"
    "&iOV_S1h*?-"
    "@|Ntu&#CvuWmr|g&HO^;`DrsDH!o;Fjhe@@=yoP0aF{zU%f>Cv+`w%A{DZuS{*x6bBX(oyeC&;I+{O~gn)G4F59pW2SCH<>|mrsw"
    "m_jaag;Y%HCh=S<|y<W2qYpZSMR`pGA0kDkOQXHRdjHMTjvJ?Er(C$9e&PXE8?iKl%&6aO`PJ#otuPW#4;bJss_y>qtQkEZ+CTO-"
    "+K{xs2bw&QbapS2xX<BauBJRQ^9pF7v>@4Wp?ZFTCNPRyN{KfBG=K2G1W+Uu!XPtTv3wT&n1o)MqS+dQ3q%~R%2<?n7mx89hS&E7"
    "^Q)<0>j>8&^Cp0(xQc%tg{^!=Uv@8zGn=G1oiUe4UZ^qdpsvVCo_6YI|J;p9=buhX`l$==)(&EJ`yZa1~Ybnf=Lr#{gSeNQLN+{m"
    "1qJ27WBCz*9h_V#HaTTO45Z8;U6&)D40Usy-"
    "&_w0R7&dD>k_C`<VME+*}DW^wodD1#l^Y+J+eay{1<Mg;K&N&_F*mUmp{`?vzY&DUya~jCpz4+8Nr{Qd^?a|3S(=DdA<Cz=lCHY("
    "HXM0WO=WCp>$C=zId~WMWZkC<q=OY{I?QFN1v!8WF_DS<joPYARf9o{-JD&L8xVP)kC;k_nga3|4PtD1X?&zdPH`{-"
    "7)xAe|w=r+}(G4db-R0DJGmmb5!dSjdI%o6Ioo_w5!;Nit#`LT+9^LrVb++fT*_$)>_D<&P@Dt-"
    "3`6uN1HBVdbtSx4<HuviGu{HAqELmqiZzFH}bnCS?x17k`i_tx8?Qe3ObIvuj#Tjc&%{XD^N%{NPWWDYEBy)Gyn2c}apRoO3$nB@"
    "$uWUb^F%?O--N~l=o|(D7jazqn)<oXcdTizi>n8Iyu{?i@pN(#e?PZ^_^<UU~Jq>?jt<CJ|bxxYMJ$GZ)Ia{4LXY2I%8Jly@SaW;"
    "r>G8dDQ*E=q?L_`2#@E=+-"
    "Z@uX?)2zs+ia{&_BFRY&6|&$l>2w)r|V7NQ`=9*H^w)|x96O+pTG3PqdlkbuWYeBdn5bItf|q7k^MP;ca0=xerq~^Zr>z-"
    "BR3zJUi+`i{g=(#$>Q2n{@-1DB5x~i|Fo&KPuRyyMzY3a-e!(p^OTXP{m|Tvv24u=ZZ0w%-"
    "x=N9=G=3u?(x*Q8*5I@%ePF{J0ZqqZRJmFbJnO^e?LC6?s@yz*?uZEox8pDU*G!QzwTz{<lKpzWWCwFkgcZIKOtj&&C|A1>u={x#"
    "`4UqT)M}J^-"
    "kGNudzKk71<yEJ9Fo9T%D}_*E46=OUI`7=Vxrp+L?9A`X|lb9Ah)K=j>~A<{8`ZJtX;C^ESuQ@jT;%b!T#?wx8su`T6KhbSu7>wV"
    "!pytkc#xd0&6?iGTh4>#-"
    ";P?K}C4kL~WP$2Qr1Y^ANorp`Y$W%jYuxxA^zRypOd$tP@+uA6P0?2A9P@)?h9IJphW+|1j{AzST?Z)BgC`}Z@a^V0b28t3S4%`?"
    "X5<|X{p*v9;mIK9Ow8PmCYnRGvS*3QgxW~psY8=qPy8=c+<$(fy#WKPVSoOi;x+1K1&Zz__md(zxZerD$Wn47&lYhr&al0?p#x4G"
    "@q*hb#=c4u!>_j>xwoxHjIXJho?Cw>2U-*?{kpQl$ZUw>-5e8%RM$=cI%PFSDqYb$?Z-"
    "TBPzbIF~C{q1MQ>8!u9?MeANSi0UB+|(M=V_P}fb5A?n|BAk!%m0#nZ*DU=cVf=&cFB6P+1sayoQ~=3vi(o(WpkTzPp8`3iF0?hn"
    "2hDyZ|#lF+gpDkKAp3ff6BR~_j1ynrsnOBC+nRMo6FeR&-qzyZ>Q$X&O2$lv&N?Pb$ajj_!nkuL?*V|Ih|ymd-FF>C)?xXx~I<im"
    "u)k>9be<5)0&(c*<RDR+kU>r3CP5Hr>uWc{%P5Nch+B@`EOWfKC%(p*<x>Gb9_7JguGL7Pnvh)+;g}1EBpAXr~6;_#Q#mUxE_4s|"
    "NT$V|5YA5hyUt>r{vrRcXrl;o1U=6?t{DCd2mPb+s{6@>$$wCea(zd%s=75&9-Nyb2js{?5(YnEjQ-"
    "!^{3~Zfzv(i=Wp+a&EAYo<?Wq5vY#D(Vtgb2gnjunw=<{n(*4aoxZ}T}yM3O&-"
    "rUR+_PW3JM&9<hskb;KdwSkP?%q~(&enSQ+{v+>tnCMP`ES_zf6=yQ%sGGVUtA|!clz8S>(Az%lKU^%>jXa;-"
    "x%M_J7Ir+X`Ox=|JoW`r*kTjZo9K3ox8dIsk!^7%dNY;-"
    "9+A2_DR|QmiZ^_KiNmV)l_tYKYeU3`;4_tJvaVc5AOE=K6^cJZ#&tiu5;47?c8j;jeI(PcjnYSPL2C5PRf{^d&ZjE`7_b!@x3ih-"
    "LKluncL0AC-OJf<(b==e{t?#-"
    "J9F$q}iwC|NXVjSewk>9Gg3hG;e+%$(%huoxhJwY%|48&Yq5K#J6&`b5B|4)P4QMC!W#gT=|PzpR(rWtm$ng*E?s+{WbU2NVb`uL"
    "D$)i&t-1TA={j>Zay-dw|m;%);oLL-uvd7v!h#cPddG74`=0{JvOt}x$B-0pWM&M`#W`HDt~t_UE`d2+3f8(C+%%Ie?Rk-"
    "{r<iE)zkXd*52NB_VngkCTmacgU#B?pBS58>#SV2^=WfwvNvbb{GIvfc2iq!ZFB1B?Rz?L=0@h^$i$poCYg0g_VzYp=KeY-"
    "%{d|SFV5QB&tJ$=dpmo^&g_%6--^<Cd+Sf+Z*G4&ruTAc=El7JnaSL_jI9}eZS8-"
    "{TAO=G=WfqEt@nF4an3}}&a6`~vi3>wsoc$cw$}DI9Y1ybGxo<bH`Yt?x8m8X&9QX*JZECvnJuQ6N&Lilv+OiKAK8fQjO>kUj&J9"
    "jkbBOo^VT|h%fEFR|2<D!pPAP~pRxa%hfdMi5AEihhc?=NXtk|}rf)nnZTg`NCm-"
    "77RL;yA6La%z(m9(Ct+usJlDDx9&zRmT9h=PG&SkSVXYTEt%-"
    "Pw)#=iY_(|KupHf!@Vy1859bMumIr^YrO+QDC2@87)b^t_3E(K%aqK6i3#XAfuRs=b^xXMXJyW}cM4KcB2WyLOVXoAFn++?apTx!"
    "w10+KlO~&zQZl4xPKX^~V0(y4$lR)<213+f2_$=51nm{?y#rjJ@nL_Vjo5RZq{~SZj0rx!DuzWmy~fbpGy~t?X0h{d?A(TtC@nCU"
    "bB0S^H4ioRm8|b7CKS-qiTcX>v1n*UvLH)+X7TGq%^KdGq<nNNVqm{@tvJ?5SCkYi{Ii=bp08spsau3V%M+{#9F_vc}eo>CE%?Fc"
    "D4j_wtgQ`K{^vxzk4SH`Y5XZ+APl)(K-%+w+XgjK4BtYwk%~s6CvOHNF0J#@|_Ea-"
    "V+Y*2vTrG<RcOw&nyk7nzRljBajo?zvv~c<S7ZHK*p~xtlZ3nR9~A);x)w7@J?~EZ(iZKX+zNbnA4+RBU=|Yn{{f+xKwdjE$`)a+"
    "39Cv(C%^>v_|2vpG}o`B`V|Rjq$c-"
    "sGHo)>*m#hV>`%H&4Ibc4OZDRwt}`#vHfKX5Q?U)BE=`HX;)_JL{5tOzn%UH#v`PG0pSLjrEfJt$CYc>FD<8`Pjs|Gw4+Q95Xp@H"
    "ZmXCi0zE*jqYZhkbBOo^VU9puYdcA|7DTu;dA!?#U4Iq|MJ7TJ@MfU_a9zy@8Rhi4{tvEaOzy%RL;yA6Jz<v_QR`hKAisdGLyWGZ"
    "Ft7?9J<!ze%Rcpho_xA>#uJ!F}|^`pSitOI%jjnsSi*7m#sH9FUi~=+n9e+TyK5iyoubs7@f0)=W{35IcJY*z0I|z^0LwCy^@^Sw"
    "Uf*Velor>zPZ=G^zbSAYcr-Zrt;HuPnx^Q&&=Fk%gx@NHL*XIH9aF)C(obaXEXLPw)g$lM(b(&OY?W~=Ef#wY;U)L(ahZ~x8i<_l"
    "h#l2XQI>Nd*_mxzqRee96oPqd}nWN%jr2AS!ZPZ_4Vdv(!BZnWXm*b0-"
    "u^S$xn}M<!sM6X@7t1bUvr~udThg{dCsU=)@N38cELl3_5>qPb7bX%V+KG)6Lu2FVCHb&5mxJTWX!Xv6=lP^ETJ<nOph0bLsvvbY"
    "jk2WIAUv|CG<vy8TJp&d$qcY|fdUbHX}X?8MmojFV^nTehFshHjm5Q?co>t!+*n?|VFXo5|Ca#7Ne8>+X+dbEe|+8Jqi3Yi;LE#`"
    "0Nb&D+bK$lu(DUSoTU{Wa%ipTW2-&dE&IoX+=aoDiAF*~uew_eL1Hnf-Ui|IR#`Gd-V25`820?~G)#HobH_&p2V-"
    "8FVUtZssgI9ovX+#rHC{b5EIb+PbIj^DjSfou=!NbM$XNvcuCJ*<|~XRkmi%&zQ}fdt|kl@rki~Wc!gRo3Si=>yhajIXq)}4qa<<"
    "KAStWFEVH6k=0Mg^_kOarH`zBPKKMiwXbB$jeY60Hlt_ki>{Yv?PP6lt=8KdpIRpyoz5of&yFPPPK<AiZ=Rcd&F%H3BI&G?=5F#c"
    "qZ@0vE#~vK)?-=I@noHxnVLJBy}ggWFsttM^qD((bNkQMqBEyPeM{&4y^P6ON&Zac-"
    "t4n{we^{~v$HqX*v{VB!p+>}^NfwTWSzP7>6*#N-pI-LMD`Rn$xn}M<!sMA?X>;N9@*{x{_ReD;JlZ;>}9u~JV((!T#zx9mdYkPL"
    "dUTM{^Svy=*CcpGZQmb^)hKCBMA)|emLS~TBd*q2Ex!_Ld_dx0uNESYeEAC^CNg7w++ugbb@s-"
    "7*FpB!3Y47H3XTl2n>f*B=#VYQ!*W=X)H)KaFpaSEg5&7ALv17L?Celvr(u}4X1G{u_80GLnrlfWCEMP$H>vZ_~2UV3EP}~F8}V>"
    "_t|%>v2{oK=p89DcMRos1j9%~2X`dkJM!TjQxQNC06Q=g-"
    "$CpWq2UrD%!~y|I*cdDG`NDAf`UVU7@GA*@iW{YYaw?Wn}#eiG#swgViO@Y!!Rfs&_fgI=-"
    "?pE&1e~$3V9Mb2{)6uSC?6<Y?y=sE{NegE<!OtqlQdHfKSB&Hw+E#?8zA$#vv57gNZ5Qb3=s7>D|FIGx!h)4=E}-"
    "fZ(8^Lcn7q*y2FoWavbEI``;1CjRy0@nA^cFmnJEgKj8@NIEqT5D_C(j}HZ5*rXB*yBRyRVi_G`qHK_?N0FPv<^pYG2*hC@naC4#"
    "7>Y9_#QCAL%(FRQ04b2gMqoxj42}?EgDV>zD_;9KHg7N<vjKzi5Dln2I?5Y%XpThVgfJooCP2@PL21rP+BlU2V^b%FG3rH7p0hpt"
    "hXw{<><}jQ4L&g*4)n1R8^LK@js}inV+jm09h-?4l2c<F8{BhtqA?R-"
    "gMks#4^MJ4#6VavQ5;b5z%W1UElnWl5>9*@nuuX13XEkQk?}VZhPWUwm6nc=&@nH9`7}o$lQR>s*=WfcB$GssMZBak6h!dAV6bM4"
    "=AbM?GCu;E7=_OagcB?olVe>UA4~;?U}ndsSu`+Ks7UNVB3C3gL5`@J9~w7sWI86~?66M(0>Or`MVS#$p(Iwh9z-"
    "K2bW&sh8iA*{$=Lk(;>44l_*K6<w?FC5Y4dkxP2D+^-"
    "x&<=Oib=%FhOB>XC4BGKXNBK;CL>+Ga4EWjr2Ke2MPvs7!S(`gkw`Bgu@vlA|nI94JP$i9I*k4gcDk9!e!tzb%2VZ!O$pBa|Y3A3"
    "xL^(BqGR@5RL<bqa|%N3E>0W5S+(FC{8hCDgqV80(c;OQF-RA&A{}8ZlPpMk|7v?`J^_?84w;y8BI3{9!4q+1;j*Tbboi|ES(T|7"
    "~t{fSVYpPfdV6;&WAn>n*m~BH;a#kOUfij2SI`!MQ$>+h0?J1g9(BTUFl8^j3gK$>nxq4GuUDT8-W>t1MpC2c-"
    "6yzYs<qjhGHRuQHUTK#O*ZWN_Qj>n+yvhVmt_qL1})A3^<60P3?m&OrnvYsby$j3Ly;-"
    "p)k;6BQ#2pU_iMfpfDYqi5HSnd3%w2@@6Q5X$+Yj4-w;HV!%+bAq@oi<i_VDNtekMM+SDHz}V`~_|O0(0hoqI=-"
    "3m%e8`O<Fr>$ZO2S|q8Xk*=B$dT65KgJl*h5r4GGMT=LAec&LR7Ia92`6L@j)F7C8Ht%fuW-!@iYnGG_4zb3Ope0GA-"
    "=^fQ%qp=;AnJvC3nK7dok*V<YesKRdFMeZtQF@LgN|54>xW6YfeMy=#2ru5fx+Vsh6wh9SrxbXUR%JHW+vMMJ}(5oVx38v^FTSd^"
    "jMfdh&Sxok`ji3;2R9&marPOv6L!jzT}aRcF8MMONga$#U@Xl!6g^!SjIa5FPXd?E=cE{Lf?^qes544Gm;g=XM}DGT6}JI+Nuq9&"
    "kZOd?@8z?UMY40AFXR0sxVs1WclQmGX<S!8a;DR<@k^@OQ$JPbk(2_h0p$0Dlpp${1|K(GuO$S(s!AT<FT3Yw46BsO<!2}5D;2YD"
    "jEm`UISh+BrpJR1TiQXq?g(2a?qp@rRFa~9$;8=w$D9-"
    "wkK<4t%(k|``%c)kt5eJ0IlqX`k4nhRqT&sQqZ0h**46NZL_9viU{oYs<`p<E0Q%uBqGoWgG{N&Z6_b7T-"
    "hrpH6YL@}wDNCOVXVy^KyNzgDlL|X;%m>d~dA3c9y0D~kLGIX3FAWS+9g?MZ%TE=D|NciB?SQf$L2wkIb0C9$7>;%%{Gt^KFqvPy"
    "85ikie5*{)U7>X(q=Slz%>0}djLzeacfJhuMP@yC?0y8o*FGU8x8D=gzpS8P}zrg?Q-"
    "uIuld(!yb2_tuh)4TcPZjRrL)4SvB?i_OW-"
    "~bojo$p2s6ad=C!!iPC#fDr4h{%Wv+#tDQIATpYI65>KVuHc)Xs}8h#C%4}kS8G=Pb;A?f*=>fs9<7tB#`nD3aO#wXM7CDNft^>i"
    "G<-`QiHhxpomZi4jM1?=0ns9pg1@>?arL#k{F|6)geJdatMg1&WAY$CKe_JQp-R-2!a}m-"
    "DFPVQq>Oxl88@*;z|zj#t@lj5r85EvKR=V0eW!i5zw{f5hDX^fI<X~<ZuQVfQO{Xu#gaN0D2owYgCe|qrjkeo?D^=G#N1_ghHdoM"
    "r;(9VBokIP?(N^5FVMaJ@|JACtwO7%8!}QK&O~U0}jVxwS{w&pkX7Ui9Ku}13S^cDE;IN4!~$SMl%G2T}L33`(b7Y#}K3=Uiw)Cl"
    "Ot*!8ww!FpaXscPc$t)6S58l<F3aC^a&<`1er}#9*jx=Ya_^ff*~9evN15=uSi5fFq0ugb#5hA<b-"
    "Yvju2y^>8#cL{jGb}``>cUYOD8TjomYl-xG-L;rKl`y$53u0*DJ7PsMmR9t{QeghvR8=gG=KP{s%%Q2{-"
    "m^jHj8lOkcTkx`OSkrND!-jgv%0cH$40rEO@lIWmW2FD@@avqKgCMHY}s1OQ6gXA+lhGS|1N&<3-"
    "%xy3yqmV+dX{eAvDzySAPFNuC{EczU0?;7=EI9;3R1ZSNz{JADK)4JH4FW<_$W4YDh+*#sn4rV3gg9)*tN>8NKsRzicw}IFvunme"
    "EW|N3M+3_a8i3={WLW46I8eNep;T`J1J0DGxiCf}Ls%J_q6ukC2!%lBqDUH=aU28)reipd&W_%x(}Q%#)5gW5iUWryG$Js`fTS&s"
    "3`~wpdEx^Yv;k`znjs)8xG_Xe43%&^;D@B2nHUHYDj#PVgZUBP0-<7KINb3-"
    "2ZrbqXq`m?<v}7Jz>pp{VdvOkFA!ivL@<*fi;cjH%*>C$5o#_vKj(<k^1FBaBks+cyEh)+%SRl)cLYKRAor33_cAdN-W!STWdl-Y"
    "N(T-o5{(Hzi3eckV#u0wLPMjtQIQj{qwqOl3}%KTq6iGkFeN_WqJZP`N-"
    "%NMU_gaX7#eh*oY4@1EtJGUl}u_dH^87muxSEbMkGb!v>oN1vP58rIY4zt080*~b{{eZCKe7&8D(^c0|^>U(1EsvQrH`!53qHIg("
    "burGiDJWA_c102;7hwnz9UD^Os{GgAwEb!Va2{q*+A3f#R8zY*gZ)PIF<5+C`$Fqqxi=q%mP)rx7B-"
    "z;O^zn2ybapPG0N4~<aD(<U?!h%_)TRWv>~$$+FSj({j({P^SzW6%byaVRe!>^cI0VN!|2gvX*lrT`NsS0>0Za5UN=idhGTJ02JU"
    "7+D0FB`FU^Wtv6}95s%!u8)8Lq(F*>0w?ohaD*5e+L(LJ+Q0j@I_JK$x%=YreROgkfnj(bg6<m!jyG-"
    "41q>40M~x6N0tLl}Tz22U2pkM{+cr2rQ(B05Mnz75EReflYz?=Ph#-"
    "$7a|^*;q&N@9g>Li_Kjokq3=f|?dE+4j+khm<$XFm|KuGCEpGoi*wE`$kT)Qv$w}N??g&YzHl0!fwc0Oc`LIQYFVImGVnnG>@H7-"
    "@XA7CQDI2ki`W)BY{13M8A2S+!(@<!PRg)|^6WB~3+lVM?aLh&}vXgq>-"
    "6p(qm1O^;J8WRq58pWmK{Fq=mhV#wtfr%J0Jz`u;syN7NM35tWiz5RE$Dn6y2!mh>?9lNr7MzAm9vdsl*dPR5S$J?w0VZY#@pB@`"
    "H6#l+CRD70y)k#s4}gRdEP_=CmXe!LH5M2q>{?642oMN}7^q^E>w%N`F*rg`j&F_q{J(ks`v0i=CvV+9JaT_1yPx9sk3a<7?~ah3"
    "v4$Hpfc9}r+G>&DhK0ri4&1=$Ll~i%&1L$4w*v*K6~z{yCFFGo$FUL$haHuN;?9t%3C$p5;r^6y48zt<$Pf%74dw<Q6oO3?@I2)v"
    "*1h97#2lbHB(NoiLOlrK*uri$R_NRWNJt?!fm#sz0Vx8klI%!h?93h(fru2SVk3YD!ow4<+;Kh%A`iqej!P1ahlCf{7)pf}83<_*"
    "7&JIe90g;Ch=zwi$09hbB^ZQ(!n}kTnRO3LV3hK-Nl+Sa_(($nG;AcCa0rOlwa{67Xam+b)Dr=0O@cf&cD9TS5*~{N0USb=2|Oej"
    "gZUBCm{73}4tD|oNhfZ4?sFc$p7Y$}We7CH<FI5$5)6@bmd+7?A_l712+YU{p#gYkXkoR(g4ec(<qSnbF2HB=&{$5`$l%b>9GSwR"
    "Q;DNV-o}BUp*AqUBO>nfVCb)S)QiHC6Uz7$jA;VK-"
    "~rGY6()8rqEWU4g8`QW6h>n+!E%0Zm19osbx#?XAcGk4MyUNU<6>ffR7?beAYZ5P`7p_ma58EY#A7ESMiSS=XYOKBFbarNHp~bbr"
    "-~6wx-kTX;KUXs9LFKyv51#ch5{1<Vb)UiLJrC@B=e*7S~|t7gTZ*X#|LyUl%xqVGZ7fVDiX(XU&b*qg6x7p$XL!AObgkV_5cWyI"
    "1(8F6-"
    "r{2>wy`WnH{<@FaXXlbL?z<C3S!MKd{m7g1!HN3+Ap|kT`e2;J^jJFcMvm!U>AP3&LX;_#`rNL9QRLM0i1RbivTn1bsnpf{+mi$D"
    "m?E3k8P&5gEaO8-xd&J~ozbP0G)Vwb+Eqzya_82StOlg$uIBQA~m0*zlC-@u6@sK24X5F@n2j*m-"
    "zhaC~lPB#??A0YYJDaHS__496i9wXT#C^n@Ye1);I7r3~kSsS3dvCc(>yq*mZ$=tR!Ww0p`%BqVTPh>Agn1bLD}Ks2TE;RFMd01i"
    "#fECV@6IRJ-"
    "(=Hu}sxrNfO_XA?05b>!1<0LzhU{Wmt29N?(Yy@Tm#K1s!IDLKkwch&p2p=&Rg%bc_;Y>3FJS5Q=3#qArQEw33XVTe?N*t7`qkyK"
    "7p{ZqPAc25!aKeNWI~Um~TY`b(*q9iiF%Saz*!Ze<_H<SNAa(=~Y2#vkG&U3mRGioNoFwTIP6Ufh@Ysonk<Fg5Aw2M>Qqu6qWHEw"
    "WH-"
    "^BF9vdnNgK=nhEaD}VDZs=)xKicg0OAbE{AgwigdrQl=^YOU03<;{W+nndM@3>+7LZv(@Q}uWkV$ew){DJ)1O$nMU`9ZNYFOoZU?"
    "g^Chd_=@phM(jbRltj>*MGCF5L0y7tUL_FlXq(k@&*Yh~yUz(m1;?G?V}wpJ#*a!ob{x^wfa%F*6a$9XO!akjusdk*L58!h;u%jD"
    "@D+^un<X_RKU!83s84Jrt_(kud;fh9n}$lMs$qEujf)6dxEF$`oN5H9??4Gk^z%7n~=5Wf&(}C_yC|f&rLES}<k+6cLIW$S?s9Bb"
    "8bK6ca-;PPs7WPbW-"
    "q3B*{C;HF3p1xBFGhdy)+9lO}g*r`Yv9Reu_5;TR}WKQGKu#ZegB0d$0E6I+4LqtZ}5D*aqV}Tn&1EJAL2Mk|p9xyh*M?-"
    "KN*m*+64kb#OB?~wJJ(EsrR1%Ck&4n^Bs27n#@iH_8<8Uk*9<syiIS~Y>ssw`p2LXlA7zp8!39G_Af0@r<3Lw+tVRVKcjSazoit`"
    "!~kYqsGwn_Y5903s!BaamipC2E<API(yL(y0fT#gS#27Jy;%qig*f`-Q;Uiz88LWWZ+AA1PQLj!P>)B<71Iv9-"
    "A9v>h;XhaZXW+E^YRV2=png&Kxts86#j|ewpT^|8KA~I==fC|;H%43Nax-"
    "l>S4^e{y8`%fs|GIzDMf>|V4*#1jnzwOL%Fspe_@Y2`5yvmWDGV<{P>8r-(3|!mc=V$2=pr^4&_0gI2pmu(f-"
    "*)B2@dE1rN`m~ON5z?jFJq4h<HMcgcblZTBi81WYxlk5d=98BZJgBNC^`J9Ha))mM)sH#zj7YEtHs&3{}GAk;cXh;D@3EL6hKlY6"
    "VUfA6?~68yv?hz~iAnkaS8!L!A#9L&q*A24<F6J_v%ELgYZ8acS5`CPt<l#z~0B%$OCx0OGL`Krt~iw6Ke7&O$UE#vw%JmK`+Vkz"
    "6(;;6U*qoY6)TA`XNyFsK(LjOWe6I2^}eXec~lBREY6I_Jm!iigd_3qym`@1#*Oh#}LD85bug7&sh@!N%t%LBmGE32hZ*U@#7jLQ"
    "mfC0JH&XLKGHUKIGz@nTVF~3<L>3B>gOcfsn4z*h2%xV16{(0wHD{4905b2M7=v2@lbsqe3u2rfDoNOp@a)?R$WX#F5Bks89{7+)"
    "Av-34t6Nq2^-"
    "aiOb_Z?$2Mm?LYS76_+p0o4a@<zc>(G%<+qHdU2Rt3=H{O7n1{2OoSIlBGo_vsV@#rTs#EPm<=t23|&mD4}gn<gBK5tK{h}qv=EC"
    "M6*)m{fjkYx6rd#$iC%|ryh33^7eUU$a2}4MCJ0mrg((ZdlQ)iIALS*nP$gWBH8y5InnJK?5<E|>fSAbNJlqraxCCM>P#qFPq=A5"
    "FsN-SSC?tRfN6V5j3DQs)j?rYeajEM4U|0kgCn0VbB13Y2g@r11A|NJ)rYwi9HI}0>7X*<92+JFQ<C11M0uCSyrDz*XoDF|I3ZQ6"
    "YoIRP#95P}|IHA*slwjaE2oTImyfA5lxU(0?06>)2Lu3ctgEsug;aCi|p#Tka2`7TZ5fB0K)F(cG!Du>0c>!V4X~^WvM4*IcAZYj"
    "@>1Po<r80qsTtl+xV0<QI9X#CefJuUaMMEkC70FFN8Vd{qM~xe@o%aX)l@Y{1h3Z@n%*f1;92+4A!wb_6+3)X?O%AyvaqyCX_!2("
    "iI8H%!iBJ3yZ{-pyz9hp13=}BZ$1&-^6lI~<P-"
    "H{}^gz>NF=S23&r~fokz`cl1QTi)gBd16UWZOHx3FP!z{7AJj%`DxJfsFDKCw82bju8pp-"
    "Q;iX>4o&C{h%Hg9h+2A}JatEWmg2IDsLS;E+I&90DS!<Dov`?7#%eW)?a!0URcw<OX8Z`vEBe43H3q4QUY|A_c102;2}JNnRYfX0"
    "F9T9ET8<!x_gz(quUz=xv<QsAO2nfToclw?qeMLK-RzbQ;CwfRg~hbPR+^8~&ZWFi0s+8yAxb1`fA|>Ae6MkZ{5wAPS7p&)5(Kqb"
    "W*_Id}`gf+LX0XBRRd{E!TT(1=>2u{R|dlKBzQm=LoL4tG4z!Kh$SKzUHk-"
    "C1oI(pX@auyZW!0U!_%iHxHfHUcwpQ$NRt0<&40=hW}gRW>h88M-"
    "tWUP=a+;uM7lx|APrJY$0nGf;480v2Tif{H{*0tfU!)43S3CQWH*1P(CB36d&`EdXY06oGKel=y^;AjjvGU}C~xK!s*N4}s74SO~"
    "ERDha~nT?@v>#GpcOYrN2#cc>LWapKyg!M_#EV~mPjDF<{K9pQ(-"
    "i2;ITS%r>FfP{P;8#uPm5bXUx5@?9W{77Sntg{GEpg0!`;DPY)%2(znAEA&2goO;i9ceNwYy}*E-"
    "fAL7*Lh@WKcwmWGJr=66#|_`a_Y{q4gv(j!hu=(P8x+NfFhxb?Ez?40}fAU!$g;GB3K0(7>uVpeCR_WMd^Tmu;2&;hS3s<J%~k?M"
    "1hIPk&_x7@{nr`=0|1=gqS5`wc~*ff&$CT`9e9n-"
    "lQS{OzV&_9TW1t2N02v$YiKc4I6<OxhXP0Ohw11p0eKWvQ198ENAMnXnI)+zbwu!i;&C4{AIxqjV@yYMf=#34jfPxN)k9o?h$L!l"
    "on!_5xEoKLzkiFpxBiKFlIv@Ns<d22Aqdc0Z`infg21D7ti=;2<bK;3E~A#i*qPa6oPRBc%J&%=(Krfj)$28REGqD<WNfYp<@(cV"
    "K)mGIx+zgVzHYrjZ0OZCmaSyh&P7FFd!lYs@MqNf#HS3D|>;DQ0M_W2eSblXM}(Q2xl}Nnc5Exisx(zWX_Ceco^t3Lgavx0Ku>@G"
    ")UhWb1<bmZ4wj==qd0Nlb|7$2v$KrM2x^sd<cU>CRLD*Km%h%3C}=qNTR^R2^9;#oMfX7qL@wV1TaRBVH!EggOmhtrLn*;gmbLx0"
    "RR!4X;i~TU`B3=O>lGLOXtY%@|ioAk4;=2h%RUF<sgL6<sKltX<>(%%YmWGqxkY6Xetsh!cW{F*b{7Ka~Wk+<OFc#LJQC`<hhtGZ"
    "0I7zQF$otjGuBaG(<eH5gI{>DG8F+d=5p5La=E7FC!;#vYdr8ck%{`SwIGY#8O}j-"
    "wPX<umfI+<t7MfGzl~it3E0m#z~#)W3<JQ0?oh;LxZbcIfGn)AP-oUH@KvkD&Roz^iX8LL7fJHLGj!wfdPm3#)JdiMx?X?Ym#PO!"
    "cWET;h`7;Lu3b1e&ld#3fv<JBx&nlCz|qbAqiyTP*^}%a2hgsX2BD~yAhQEIH3lRYv71)5LK+!wb<R)0gwm+%k0qJaG@drtm=?4o"
    "gLD?2N1!TMs=<SPUgq(Ol)}SA-TURRzKv5grO@U*%b`FBFq4C#gM-"
    "O53k_OfR4#4!1xL>OcE+!2e=rrnT=!m*cEs%P*H3FLxwz(j20rq2l$+gZ6g8|n!)4;$37a91Q}>BA5=jjcp0hG3W9U+oiGS72dK^"
    "suoQ^Q?nA~X1k1=m=O!RDngkk%iJ%Z@7?zf?Gkcgp26iH1B<GNfzjMXN2!m;0S>E8%WQ6bn8v|)%Af!P+=57gOBgTY4rx7A4Cjo*"
    "n`P@4c1IYA>aWSM|kkCj1N#EiqFakZX0Uxp^Ly*qr%tTuvG2ua}02AAT_?C+}14m|qC}zW*06+p}U>Jjn9MY&U9cMe=1BggSWE$1"
    "55jdG2Q&aJI`bYkOD`)?EuADJ{<v@NV8C;1%j9p2CE93UcNOWaf$D{*IWg%k-"
    "7*>HBG@XkP3WQgNCOo4eCm<_|naqnwm|QR(aHs&NZA73#Ge~;)IBHEvkbyP?D#Z=72JkXcDH>0^Bcac%7zu1?prIZ%3c)hG(76c&"
    "jm23F_QN79W2UnR;Gi23Bb#0$130FEWi{iDII~m;2a3;VJQ@PQp`j^f#vC)D(Xoh>l#_so&17tLxs#?bYI?-DI1uGEB!DDs6-"
    "31J!-"
    "ph5DJ&ptB9jjlJTW{k0Stsy<zv?v%#WHDp$?`y0XShISjke(0jz30(8p}&*v<n$AUM;ghRwtap#gd@eR=Nhs#TV+8XCBYO|F9JRW"
    "WoG=Lvfi7+sa4J?TJGk*E_ylGGt<Qn;a!U>Gz|$3QE+NQX|s6oLmFEuuyQZa@#6JY!=vf)Z1z1kc&nRtr$(bgYk&$O(o<yE|bJVj"
    "%}hA~L%Vd!rBwXBChJHJaEB#6(bt0IMX#y(|PwYy?nzRnq#<Ux^Q(&;yo*3@%OP3oo!S*&GCG85lI9z}Pn?B)W|dNjV9a*i2Ts$$"
    ";{R2`h|;nn_qGSOpoF9I2l8kPTP^3kaLY<k5mBbi@Iz_88jrkYo(zN0CrT!<_(-"
    "Fi}8J62M>(cb#uz0H8=@8npr=L1K7jbYrK#tEVhn9Zj#M_|+i1+9w{k+Kgd`8B%-Ffr7$9MyS9InvP-FnpA3NM47AExTsucfmV8v"
    "4xPj=1h<ff9RW}rs4y~kHGUR`VcScB3^WF~3O9has1?k=xr7CS(m+u6VQ&;-"
    ";cx+|2_lGTAdU)$X)<O6(B&dhpczmjt6uQ|6nen2nh8mhwZaQ*9BU3TwG5z$Eifj{K(~=9DJKDidCA<H%qWi-"
    "NWmb~NJ1qXn)cuUYX*etGz5lXA!EaX5(U9ZA$=^Vg)H0&zzIo7p}hgfAzd57IkxjX5Qv0CCPRZ;iJ#&#V+;HFT{C6jnqYVhgRg-Y"
    "K(4X9X;F6#5-"
    "3{6*EqAFvJfM1;B*@YXw}H*8fWYpWVWl<1V%=r1Dymc7>AsPL24Tln!&)sMY@#}b5<ph#sEbHjo@YE1mWpxxHE~w0jh(gfuQa~j!"
    "*-@0W1)-"
    "j@;zfKuiR9s7amcV}ccch!p5Xfw7cpd<=yi05gC~k_zD<v4yk@3>qp3)?*U}dR#hA0wxB+ggX>dUNNMC0oX{=NISvNv<HK%;f3oo"
    "L{EfbBk!c2nHZ{A;2JnOG$sVG+6e#&6FDj>RgDF<bFAwn0wR^_+)9v`n#x>E{#`qF@!Dv7EyJ(H*tImcmay0I2GA~cps6eb2o-"
    "1p&YX>}#lxy`j6tF5V@yJxi;sm3Ax8_sh`<d7A3V~noS3uT=+c}$1GoyovX_xc(U`o0IR%hFkVHg=y-"
    "^4|a8l?zsKq8G_MYmaf|Hi<tN=u0U?5Cex%S%hB@~hXGBT%GE4;wQdgBoTpr|igMw&pkQ7x?i0t&;zp_`8>uNYEkAk<6(Nx>?}z("
    "Wrnum%>S^Eq7bgqI3{Ln`JB-yj<91P}y^7$^^D6Gotqwa&4f?}0!hBr=U!fg6GoWAj`7uFD?3E)-se)9V1_g6jx-"
    "9d7{bNe2qbLb|{K*dYsq*Kv_xR5Va2a2*oq5QwjHcU`17TG$2y*9C{-"
    "Bf6Cna~2Gi26IB02Jja3v*=BS6mSC$RUHpGl2Cl=XdN*P_LvCE#&}i$B2pj}jx7dHj)VxZEMy#MvQ~iJ#%1FX1E@SO;2Si=B2`*}"
    "1Qe#@SvM1yUNJU?ffaiTtn{seQyvTg#)!RP4G9BkIV0c=56Y1>3t*^v$Ti`HS-"
    "R^05+)Lr2TKWQJs=$KC4w^@n+XzAQ!@{cf7h>b!1dAidYoR5pzA$wJz=khj22w)Vo`$1LV!>K*dYsq*MpH~prEK+h)K~S30O$vVO"
    "bDH1a3e+IM%HxRU+ojqyb!o8^K!?jj=nJJAercM0Livm@GaebiBb{5)RWOY&t6d6{?sVNxI?#A;cR;nhXqVtTzXlS_V|kWuyVRja"
    "2D4379xMI5y`Y#6U^|V(d+tq>;W=5D}kwVna4SDUMA9hHPy3PND#+SRfgL$qk|)CIU-"
    "<NJ`~ck&v#96*;!^JrIb56sTe+g7WC>#=jd9XK$dB8$fu2Pj0aN4X9HMpgn0r65RkNWQQ!=TqYPOC@LDtnKA3Yf#M<<gA@n!zzxw"
    "6-"
    "I`J*k!Aoz;YRQl<tCInIN|`+3DQ7N*C9+I#ixW0G}ueRVVbl|X9XZ41B3CDD;E36vXH@;FF>z12bo$18O0%@(L!U+MzwUD&UBoA!"
    "_@R)r2(#)q>;XLaQMLj!0>`af#N}4YPHAC7H9Z|S-"
    "9%}5&`8)NE@bcjzxe#G^kKJ@l$qef4>`3=58c|8!>jH2W}+njgSGfCuzk96*wKTR%lFmMnz<`fFVV%L$Hv@!yr|G3KJe5xd}nE#("
    "+v8SoSi4GXCZg2f)%mV)vm%>?vr&1daw05F3Fs<1oT>RsbqAgW=)hl{1V%%R)wyNazJXiY2C&!C-"
    "L+1dVPZCx8HujNWiydc_z@1EI!B!8&;8@gZP5;X0o?1%jYd09`%gGqYh9?mDo9i3;VxQbM{mg!8>bB&0wUyBRu>wG{i^G-"
    "v51KD!BoH@U>|H|b6_fEJ|<Npuqpnr>s08ck;mGAjjcLLwxo3lS;~6`jGrP2do)=Ofw50|U(fiZTu0(JXWmi3Q|<Lsj>oMeG4!!&"
    "%U0ti+l$yd)e}N!Vb(#6XyOMPeUWHY9xkdc8TQ(=vb}Iunfvi*6%Tc1{8ci%x}yOs^OT!@z4M(WPJ&H2nAwFp6-"
    "Uh607~93U!Z_=Z`v>i`k~RZ0LGf$3V?`5p*FW*W5uC=SoA@Vhx}^ky=+IfNX4GwN>kwTy4Jp{W3&0<g>0q&H(sFe*~2k1^>$rno3"
    "BIwJx%a2`C;El;|}fJ)&;MNSaC39LFfFc8&yqmZ#;0zm@_h%w<X+jv$%1|~;{D?Z{Q%Z8*c6i+r-"
    "d(<rm77fu!**Tr*IDW%{=}DylubHHgz6GQ1;X=T8LU0<27Q&mr#3vOCB*Qn%cAWr=Fp;Qy3F+FHuC)jdkpd0whvd|@zgxzqZejQ>"
    "0J_EYmc`619=^qfg0j#p5V+tkRKF+x1=Bwn4xbE0PiBKBGnkMm%0gp81#aMU3`6$G?97ux6O>UA5u23@EkH}iBS~^0IOII+2$t$&"
    "Do`OcF!03UsC5%0^nyr(`Jf6Kz{|)9oNTQ-m_sb&BGN$U2p>8|Ar^KssL(kF1Wlpg7WPr$FhIr(X>p`LHv;0p@u4ewoC`w8&cO`e"
    "jx<?BfZoPr<B_RB07db_Wh2IfK*u6dT7fl5GcU;)Z||g0j8YykVFiPPMiNNURzXCZ@^B#uWJAY+oJA%dDtKaeH=;5!bSe{rxxxHM"
    "ZcM0H2g98Jqe4<<SSSxtlGdmpoMT<z1Bl2>h6>fWmDm|JK+a`s?f3VTnOjd88h8qwJO!p8dkPI4Z(7)?1_^vjNC!S88ZrV0ZlH7w"
    "BNPdr5{W#+py7&QT82Cq`-Key4#}M@!eGD+CO$ZZt(zdB7dQ=$p-"
    "52(20|D)0WmRl$44L*a<DWI5RoBcTucmfh0Zw;LBq|%eqaiR0aEAsm~RCjA_bZO6c3KmSME525o7~68<i&Wg%{X(s5uDMGJvAN0%"
    "OupVbQUOl$4Wz!n}l^g6`ot0GS>!E`}qAa}5bt>01REc-n&pHjs@&Jpo}BQDf)<7&N>kQ4ksl#GEl0-"
    "yjNNxD$XACW4g*6$xN4sErkI*7X>XkjONub1U&eaDtzlv!C{R>Usw}b!gzJbn;Y~J{3VvrNL7PYq)?>w2wvUKtW+)NT|RKnvP-Fn"
    "$&1y^i*f;sdxZa6kFh_WF&en)(RU294(?o1a1%+eCp7ckH(}*ykKc;Jg9IZcp0hG&*t2boCT_brGY{{Y+OtXoRnB@PB=5}6x2Yhd"
    "Q3PBkg)Nr07RrfHv$JXy5a*UB!Qj*9G62v@is=AgJ3NKgJu*2>#+&Fkc$vWISD8X3#Z>?KzYPK%8#g+3|Q$~1p(2}!v#Zz7ZxTEJ"
    "rSBAd?!%=hgD@_JtvvjBGkcZ_X8jigo$9~!BPUa(t4newT`o$?}0!h#6W{viJ#&#BlA1{JuQFyX~Fbq6n`4Vo;D0j3p>@2+LI16m"
    "4z6A1E*scp?*Un!Klayl*)y)6ul1N_*g;{kc$h#h(Lv*p{I?G!4{P&@q(xEu^riSo&mf?{cOTbhYkRpEe%BVuu+Iw7DKJEz|m-"
    "8x3D*b!!!wT-y*=oMgR}2a*d2&&;yp$3@%Nk3eekFXjlSk89)(MU`(1ow-"
    "F*8Cjo_d$=sU>JcO9A!noIv0Fu5{P+&xS_|OKjacC_dEI6M>3j__{NfaO>$kvDvn;4mUn58=bEWx4}M6jw2V>-"
    "un9snW|nMO72L_kbU<?T)RJ$>rl)1&dzDgJbb`GoVZQw^YH{B&m)R2E{y)6qcb7=Ah%l96CkgvVS+OUOT+nRz;PfQw)(3!Um?E;I"
    "x3;Ue9-"
    "2|=*t^N=!);AN!J$dGx*M;rh;TM|*#eb^g?SQsxL4eH^hdOyHJO~R(L2vDIJ&_k)$$OsAv3mHe6OckJyH3ykm24sFHT&5Ef20AGz"
    "r!z0fzsbz>h#?imy=D?f`qsf|4+dES3(_&6Sn$N~mQ(;f;_MC>zCkqHbzmgMFs?jUN&v&gbdK#jMkJ(C4VwuPlVg+Ck>4{`S${?}"
    "eg?yzfgqnegRq8E4WJ20t7quhGyLIaph43kmThtK3^;xU6<PrII*b(;K~5F`#eo};4<5m`mns=(k}@I?W8R|N1aJp)2dK`L1~R)3"
    "d!vx7!T}e#35-"
    "==2sG5BWjreY5jg=APrRZrf(XlSll7rGqxe{JP^V=8MOJ~aXiNxn8{q*8fF$iEGt(=ER2m31Sr$Z}6|8~+BWE6dzy>I-"
    "1%ypBR6OWLC<s;$`HXLhb#T`SAP5$@P?3;EP2*bYdW^_SqgLRC#7xdo-tYWLOXr8e^I>{EfX?Uq`MSdlphf9GL0L!_D)0gfpU;LI"
    "Q;`!C6%A?Wz=7f-"
    "#W_N!`q=qs5PUGKTT`k8YYeEAj`cAMIYHjdC7^?4s58dJPywl<bp$ooOTuBAgbfB%=tc?SCs)=m25kTtN1Ch^pm!UAKmbLD!eyc{"
    "q0wzrOUmgC3%Q$*O^=vP`4QJlqD#Rl2#C~&3w7X7MYv8wLxu32Q~(@;fosA|v0kjs>l2GGhCSuMQbO7=rt^Id1R^sTnt>Y<GufNS"
    "@0n9KpBV|C3Dal#<e9dAChAlJXitJFBvjyZ$g)jm3^FSPS_%2l5^#<nj0vs5XW~0P(yd3l0OC@l%$Y{;7Ud?oJ2>KihzxbdC?s4s"
    ";3BBOUJ?$gq-8v-"
    "AO%9<#ETDoWLcM!%nNK>HZ1A144~*xxJ)!&Xv`zhaS{NMgd3)&4=aq5fj*#E2ZtUk0E{PG=X0k(5R?E`D;D_7Y&LZ;5EzMx97V}W"
    "qo#4Kb-hF+q*9&xAu%)OkY{E5<!6n~Jd5Gag4nYt54&e^T9nT+<7ZLmS#AP$$ij^>1JweCbYQHwh!&l}05k9`e;>A{REgFYP$}G~"
    "$O#5+A_1K(sqPqsj1?2ONKCBBxB#SOIx7Gbnt^>uFmw_Zc|f-"
    "=H(4t{uQvyES_VLFZ8!viMYoYEJ0}4XXWkgqy^9^dHIragLE*=TfKi0&G{hCccM`x(^^nixW>aNKpVNULOjIZj@|tnJmxzQ^Y6U<"
    "V8C>!AIXQ!$!v&v%pwFSa;Z&n&Pl5^%Dge7|+2$D)nH2=`z~=zaFD_D?EC^%5z~iHUq?!Q~Wf~Q!H0egMnhDZCW_OH2$l_B$i|s_"
    ")ld6}5!wAz^1sRwapswhMk1UHzUnt(fSkh^a62&19gd-"
    "=AfWp`j<PZcQrc*E|8WJ?pw+=QRJdh161g9ac5Z(kPKB*KkSkoZtb{#;%QIxEW8nn)_0uYGIbZjQf$du)0kNn+d4@^88$Itf3vkC"
    "ib%>Y`IHndPDRN!>kTJ+gq<RG(Jz>uQ1V{_O-sIZL)1|J`pQYBh5fU8;~c#FED@D_wsCrATToiS*}M1t0l(p29E(hMLQ(^&zC2#A"
    "s|HX>|DwgU8ebC3bl+#S}5359MW-{}ks<(tfC!-Bq<q>+MEQ26m7N-+n7_J%d73@AXY9`ae!Fza?sz)F#-"
    "HcaDM+xZ@dNP%Vm#e?I4-"
    "*ZyupF;)D0nl@7{~X<^M$w`)m4%+;0kA&@pjD#`SuKEjrdY_p09C=@<D*`x#AyalRBHq;qr4kpYGw?WoiSwbsf&~bdzV$xG9nO>f"
    "u~(eU|G#LQWc6vo2g|$6o)|2MZS{&kT^FJAjVJtM}kHQ*1_)ML)r9HnK2Pu2;Wx)3)Mq>Q*4X1922loq^fIUy4DInAR1ILl*VUAe"
    "$UOCeJ;VD>m%Dg7cz?W<Z}_E3$n}BXgX7oSuM~?(c8&$A+9)C*aixe2g9hungL7<crGq(E(9HbKqwU}L#U(erg~EVvN4@ikb$RNI"
    "JPV<eW7^0!CESdc87gqbQ#G5IxsPD!+~NT#WhG8DE!y}7J}1|SRmL*1whp`L~FZ(4PXgJDN@yjX<YAnAP^0j0X(qb?}G5a1sJ@*0"
    "hU$G1-LX7AXIPxVU6Y)vRXj*bg_^lIwJx%IC*evN-NMNU1b^-"
    "W!yf4rP&?h;*;W2LF>rJrqHBiM4&=AcCScm1I<X%7XWrpr#*To2!<dp6#yjG4F`yE0N5JoTO_h}5aS8Ksi8fGN(HFZHAHLs!D=}s"
    "9OcW}R6*lf+xZ?SOvgYtJZ}FkOdP)u#}|6wLd#yLnF~D(DnNk!LV#ZAFcm4)0<CnQP+a5)+n7*!d}K<MEX@FwGL4EdZU?|p?2d6U"
    "Oa*Q?)k{KVmg%g5fGBWrk!3YQ2*nG{K~hm81P6L89p_>6#;9UCr2(};(m?QI1K9|!#u^d};rps!7fGgRHg!8BMCHp$8^*IDk&uC*"
    "G;Qmm(4V;|5MG3^i%hROUteUyiwNO#+4{{hX0?Fs>7qDU*aiwt9t@ijYZ8!B)ypV!`^=Ua(xF12LTII_UJ{zLj0jAecA;3ej`RiE"
    "IxPdD_(tf6RDi<J8>331rC`&gU>$ts!2;PRLU4fsN}@_eWa5Zsk6j0XaFik|T^rN&ULqPaI5(31yEr~_F$OO-"
    "z3u>gv84>80;S8gYA%B-CLMSzE^-"
    "78oX19{beaJKBX}7xx6fo}psM#qAx1GlpqWCGmQj&`!xtV{R+7F@eAzT41EPoq=&J0Thy5F)kkUZWASqb-"
    "0gNI9#}&hCwFJV9WJD&8_#G0CDy6Dx<5>YH%;NaKoZltE{1Sv+V){!oqv%U8r3)2YVp*dJMny%nfbRVQxk_}#gvMi|UaG_fm1>RP"
    "{h+vs1eR6Tp^FhJXr=Yy=%txTT1G`qKwL}!8A-"
    "N6@j^4TM}0!{T6RtX@!Rx@2{m0BNPT?h3Bkd|@O>?THX|CdDYom75S1b;T^rL$Ktw7<<H^5EBk83+xzw<#xzv$@(uE2xwXD=UV^&"
    "NuMRBs|j0l{^M!j^J0R$s>H@?(av4AbB?ijo{T4|~;1Rw>gAR?|VKGv-xTcP-}X-EcE@r}?`**Tp-"
    "w}%uXwi=j3u?_~ICj^HS!}qnv9x|I^yABCal~UEUX<YAnph7oF7{Bb&wE4>*cA4of)68X#G}UDuXu52rCK$LH@-"
    "i*aMif^G+Ypa#d+9V+$=;&U4PjfF-"
    "6BSDwB35&M~!SuXBCuoAy7+vq4=_ynk}Lcx+*)TGw1e@VjSQS2e}VmY9Tm@izBjEfKVkPGRzua5soUQs$17u0f@*60#m^6^0BeY5"
    "q7y@HRH+4388e^R?RbAwSexmqBvP}Mg+p+qn>un0F`QuL(SV^wiJ7B@FHjt`M!@lBLXN=FDBBABvs)yQ(HwRMpvZ*BDd)~rGcW!M"
    "6mQD8$}4tDWDXpj*#YdK}g7#mJxgp6y_zNl`Ht;jGZeGc7<Uz<H;+0;dH+V2B?Mvi{dKL84)OtjXa_NDZRo*%xwWdioG`qxdNb2G"
    "espE(^&-"
    "#UOv){DimKfQ)5LmMz3Y(JY3xXRSej4X`t${VNYGx7Q=VasgjXf+Yh$O1>q=N*QT?b*O<jehp%MLy^;v7wEdNuk+rEh;dI$vsTmD"
    "fK=(pXoGdy+JhD~E(j;ISht{``ZCUl+DCA0;>d5egsH9~~7`}YuNLA>q(=w=vZ;Fn{=GGH1kc<Y3rc1$^j}8<eII$R>sFIPJ+Xdk"
    "$UDu|wo!6Kj&G}u$C0E)0D$U5+R987bwnn?kL>AC}T|8$S;xVPu6eW8*xJsQ@9n^ax7ZWHA6Sh&&(B&c7s6zA_l0j8`Q*>1-"
    "0KHA$DG|s-"
    "unGd8A_S)w*(>0rx~6Gvw?v^7S?SibRsaS?<H+yoNPM*eu13t&n7Y~lvXz=(x+(+F3q^6VP=WC1wwGOVmF(dgac1w0Q5@YiQ&h4s"
    "B5;G|!XwT2LT{bMifD?iN(G>|?mOkRMhez^9OJsS$X)?em5kgJi-"
    "|&&veLC_T<<mJC8@t_sN@>cU!$38BzcW5oGvxNnAMPGQJiQ*&^)^BW!Dtd8iyWk9}80Ky)lZQ?IsCL3f95Kg<;LKLh)tmv_zE=qp"
    "MN@*6pE=sZE#EBb!>7$nqNKQ>dw35TYt&rEAkT38;|B(ciU^{8}4atC?#hd95#$E?UoA>j9=XLIusEBb6+zQl?>U)d><YWidh9RI"
    "f|H8qmEE=t$ekS_X&W8%rv{x=m!LM)t%(4pfBTieh-DN~URwHHD~3*_g()RzYr(^1F^nuCu&muCtUbuCr{_uCvi~jx>wnM5Dp$Ow"
    "`MzxynMyZ2_n^<xH1TLAT8mwsEm~Ay_u55L%}js-"
    "!@doX(cpqlPM)E+wqxf^l6@WUl}NHO=jUD3l@_(>MvJkerZy*T*KVx54#_5#{x&P`c3s(^W$XMRAqrjPsaMYn75cbIYzpj_Hb{4X"
    ";bVI=H(qY)ZwKt<!E1jU`Q1-5%;#-*gE*veBGaWbah7wYKGj5LGEF-MZdu3=7fU4QzUY>2Hwr4VJnA>rxX;QDp!YS#-"
    "vEOxd+cwMOvL?Etf@!~{x{L?s)eBKWd_BwL|31DHhUl9NDto2XN2Yb5;iMsN*k(@r{7Zi+QUp-Nfl*7aUvOs@TI4CFUj{zh5fXsH"
    "|RZZyH%r~}d8sJlvZ#&}HGwMw<dpf}2swye5JOwcw{bf&Z6%d);Go2d^)Cq-"
    "8!eJjRmiZvhEG(|v)cdBGGD*#cIveLB?yvDeJ_IFcc;3mu7B<q`c)}f}0RxnC!rC(%mlL2osk<$btc;&W$*n1-"
    "vK_6T1H7QsixCsp;*$TZi<xEpjq1Uo=(5;wGHL@TEGC(Z^R}@QDRWeOe`yxbD%EmNK0uCSg-Hh^^b$7EQZ}!E_h82ubTd5UC+$_M"
    "G<-"
    "C_mlR#^j+m?!vi=f*kiOzI3c<ELZdc%s*B}d$bQd=X<M>d)}TMRE$$sC$uB~d6vcCPmt%l_Shz*{tPiy&|5kkXB&R72K9akp5Ycx"
    "0<JMUl5H6;l=sv`rG7=`4L&Ruw`sfM$d)IsaCS*A!blI?!BEEO}BTBe!Gkfv9R-o6agIF!H?6$ny;QJW)T-"
    "7teE>SzD<US#(BtOt~~gJ<p`ivzghABIsi?MQ2=myb$b*vUOT1Mz5v8Jmj@Tx{oK?V#%u3?Zh<Mu?wP5rL1)8B;auJ_k5H;Uv<wH"
    "q%N9XHNkZC`Ko`u?nI+Jz0{guH1vF%iM=;v5p>%Mp-BTNmyIN;w@&xP=(QBFHejiuH4=Ih;S{LKt?Pm)R4E(NS(WAeKG#Q|D{D!1"
    ";&Tlvn4-3_@t-TZN&rtUyH+XLcee_}?iITgqLPgXU7Bfy-rOSeS}Ne=Hj&rLr$su@+}R>~p_;Ahix5?<TPG7gF@K*&B%h}_QGQ-"
    "4nl828e4YrjB8$!xk8Hc9DC{<5Z{%XPLR3<)#HHI-2(8mq5ltsM*WS9XRSl`97J@rlEO}BTb7)-`M4@V38^PO{`h7lzKVLJS-"
    ">Xi1zT1@AN}<R?0-xXKP7@4p3sUS|5p>%MQAxpqmu_33H>HTClf>OAwLwpBoYKjvnr#ssP-SC8;Bf!<1t|Xl#YplC>|V9re1Qmr("
    "!0(yk12J9-GZ!oZ$#ZTNpyz3G}DT*b-"
    "F5|>7>ZSfnN2kk>cZtqFC~znyu@LC{(R$BY2OyzZZb~1)6z5uU;T`sg>phEjNoSoX2UEf^R`8=A;O^ZH4HJ2*k?*swkUVimpoaR!"
    "rX-DL%5%ODC&pwqqAWB`e)JoyC7IblD3<EvOgPU245C<O_8NK3c0(Yr0!?l{bpgB+(h-(rhcr*6BkLO(&t-P=ZvS-"
    "Z(|AwW*;kD%ltj07?IQ5uCh8F)!-"
    "Xi{vh~uC}r+vT&YW>dLzXS#_f*tq_$|KzG@w6+*L$&?P6_is_qT&pfhmic_1`^+gn_*0oW{`0vFwc(Ej3+=`}4t*EVBe9bD+nHOu"
    "eYn6I&O<ycWg7iwQYuHJ_8kY`QAvC23U2?*$n4&e(d}^T=wbn-Mi>PF!duu@O_YwfUMAR=)#Y=?Nn^tP263(NwO7-"
    "X^LfXpSbfs;Q=nQyiMsHRrdM!Km_^o@jnU5lhqR5i8t?P>@RIOX5A@X;t%WjqAt(q{}t&J=|U1Xs=z0{R{3$p5RQCcA?d8>kNRc3"
    "CLq~4SwbR9f?D_(8tqlmV2@}$<LhN37`t$S-y^Y>DOzSNd4m3!5csufuXPcN4y82uKcSK1~C6$qDZ-By*>)UtDr-"
    "@5OcWZSHvMd_qan|ADisAOXVKlHoJ1h)yY6}MIPw%!y~E3#;wUUuy^&EKZGs&_@um8hfws!Ip0D4SJ^F1hxWOyA5$5ob#$g?eaR7"
    "eu35r!)U|yAN-d<n67vU2jEgWnC)KncEFnsn*oD?3!|>mrQAmG|-"
    "vL0<9>UQiNXKXS)z}s%Vmxjb4N*w_{&Kp=#YapEtjkLHuQ+epxGCrnjQDvM!Y{9<2$cz6DwJt|+Y#qE#-"
    "<>a8zAuVs5nrf7}qo?3{dlZ~liDGF7}#u&`|-Qlu3BzcG2tJc+4-"
    "XS{U(VAe2TXtS)TO%7|T(Z$yUy5E!_107)yQdb5YnW7<cI=BNR4D}u1Sh{cvEWWgcj8W=b+wguszP|QRw?NgWYx=}v_>{Yyfh=UP"
    "RqTP?Jb$2Nme#(>EuamTGvZe%Ep*b``zWjyCiv+-0!lgRw^MpwN#XJ3(_cUD@13^OOircbj5_O-kNG;_4KAKoor0k7g4BM%YggcZ"
    "TY)<b$8vX)>W-"
    "kLU?MasO}bI)lZ6`E72MAl8n$6U9qA|?(tjkMKhmTv_+^*>w2kbEqVTTkKym>RMV?g)K(r#C4{G!ty1n*Hb&6LRtOcGT)GvSQi@*"
    "7>XuB=%*QEJm763Q-"
    "P_pjUe(^)sixm+)w)!oHN9M#U=Fw7dUI0RR*24smuzZbm7+U$r^=_aiYQ7al_rTw3O4e)Pq+8AqUrZp^;l%lJhiM;<L_&us>|irB"
    "mrK%ESpk_E;-Tkt$S)AmZ~=G*cVZ#)~HDN-"
    "S4scdv$;9@3(4QDxo}G6U^Zjq+Hroh|YAEY=qXAP98xIh1N*HEUsWeJ+!Wus@58DelNGc%X{^5y%kj`m1v$`zFahCZrOFE){=m(U"
    "Y4ydMXzObcdBUOQ;VW>vTet{h(h(w!!LjN7b5rzd-a8N|3a%Ci!4r_UbafQTPc@gD?}yLOOj<%N+*T8Q-"
    "TysTS{3Hp_i(ZqDOwOF#Ic8(ezieRV%M(%qwcXv`Te%E9KI*MjDqS%hs2o_pXV`*2v>ii?&p?X~({ZN(z?vy;8NWY{e^u_DV^<vK"
    "0sO%9<^$QqnE<lcF@i2$y8aw&;p=vbsA}wC<@zQG|MET`yH@JpL+W_*JrbmE2#|)<(+WRgzqqc~`wF$5x2WG?#2jR~4blcc;{+7H"
    "3N-OLA<-UaC^6ME!k{0luhHUnI0IlH?b)qItSjsrnXsS&pp`l~k@?maQ+HZ0}B~Pc6=tQkF#MrK+_@KmB{P3t!!-"
    "R|~DELV0y#Ufs#1DR(Qoa%_drfM(0qm!g-"
    "hfkNw^TAVGVEQ!!d)tdTuj9#P9yhb*!5qnn``5IB2JX(|PR(9ps3L%5nC{}2PQuJD$+?^^~_0*y$LOrx&FI6c`R{Xx$L|@#gFD_e"
    "G>r#o6r<dMc^|BmWAu8EivMF6<>n_~2YUJ^$#n~d%Lp%0GR8sWn@3p#rZ7W_|wxSB960eoyYbEpA-pQpBl@zXCmaQs9m+x9N^2}3"
    "<qExkMg(#F{-QSmJ_DdR3wW8Lg5+_eBE7jdq@5-"
    "??QoVYqluj;pr_@I*RXwz0UqquV_W1904trfIURSoRzD^adYh|VCyOm{WTO+%xmu0I;C#$<};v<%m+O%UYRqvqR>t*wLu@`lbuNT"
    "GZTUn{@u6kLHtq`60^d;M}Ri)_iU9?6vqAe%2X@#hyV71?us`g78QME5sN-4gylS`}bR+go0jjS%ol<iQ8uHHoxpIS)u&<atg-"
    "Z{TFDDDk)ui6{>Qi+qNYn6I~B+JsaH^}x4vMF6<t-"
    "Ej4Q;V~us)u&$i)b|S`!Y#?S=GO+FhZ%smkF&>XYN*(rEP*ab4jLbhf?&TYe?C;XAz67>!B5*P?D8@U#{9OultwF^~+n)JWJDWxg"
    "1*|l&h7EE_y9{Rm#$L<)k+4*lVeP>i0$+ys;5)Y-?RA@y1qG>Trv_Z-o%9R+Q~fI(c$;s(Zvz)xOu#fNzqf-qg#w7gZ?5Q_DAXrj"
    "=zmwnC^^E2>g-`7T=bEMl>TdT7VKh|V~_H_PVDRqyJtRN~F8d~;`7S(alfgm|^G(M1>Tz8xR2RQ1pb(P;Sh6@vVVvUT+<T5<Apt-"
    "GsUmSYplr!UEr?NB=T_}!`Q5lh>Sy;QQVeqX8judMr5w)I#_@s*ua?yC2#k;ki-Wvjb<AD%*<dBoDTV=t8k4!^hP;4O83OIzR4is"
    "D&XrQXuXvK*UW&Ro4*+G5qaXx&qkQXbl|FG6tg@2h0@RaNimSGD5g=~{hP{iGaQA;hZ|i!OR8J0(6vsp_E>qLSe6t9$v?Wh?5jRN"
    "||JR^L@WDaTd_@e<mh6ur9pc6`LrwqtL#`}-P6e@)qndMv`TH0iE--_)xWhf;LmE?V`prK(F`glOB}*DCI7h4!^o`P#x1{%ad`=B"
    "|2KDxqAps7le7yKiTnTAVE>wP}@-Z29{-&3>KGzOE|Dqu*UQwnm=0dbzZsm$Fu+ti{=KQkzyO4Ltt)dKG+q-"
    "M@a(UtikhSz33?<?CBnT&?Vja2Kt579rH7*An}EL#Mu>Y~N6oQhY-tmAfp*R*2?mMOD=M@El|<q*{tfdcU{!@~vfiYhRk@t&KX|R"
    "qvZ}39atcU9|2wTS|Fo$6ibC_l>>$#<G25Uy5gG(%qFx2$#_6Ufo5jp0=FSr7xm0?eCjJSNCt~OY<zPzUA^wtvtC}S(T!n+_kEnq"
    "EvP1i&%aodRrgAtrc%;>)TrKHldxlTX|9{Azoc8dV3eGdWv#V5AE1nt^M9E=(m^c?NvF5w=e$fwR(FaPf8`kt89l-"
    "^yMyE^%SM5ORuH<dq=0<QM7j)<vU98j<QL2S&q%P%2xOGK0J@Nqc2KTOVMcY_ZQET%t4&=v#qE?sYLVia%qB*Zq-"
    "%YBCa;PE*qmF?fUfSJI}+%-|3QfirP|gmz-"
    "R6Un`y159az>dTB;>eV0|OTi1Ie@ve8l{9Uf@RZHIGI=T9)cle{ZR>{7uveSF1blp0c;1dtHyAL?z0a07(0m0SOubS!|_h_zFvOi"
    "SB*~HS4_c(UF8|3d+oTcP0Ijj2W-7bAMM0Mp=+4WskwQikEu&lh^LuKz#-"
    "FrHv5#KFU_qFOB_GqqEvVV`s^e(Gfw@zoH*LxlEUPr%ICGS<8RW;Q+?$Mm(>ngjxOGMX__qcz(59IH2_4}ayK9lLbR=x8di&$Q&v"
    "Rk@Dblp0eve)}vrnz3(9j%(`9rkFhRao`AtUPB^{Q72qYOYsumz-QB?>YJLNXxt`yQO=ocfCiF{`vr%e8ANofcgg%_5sz`s(1dQx"
    "mICS@1E+NOvbOjWD;5Ts_v4LtK>Z=KNe|~S7o<!PxVeFqt~~XMAp5kyX54msor6a<}81xvg^C7YTY`UqSprj_CZm9&>@nOtFPMUp"
    "gyS1AWEvT>$|LKEqNsAuMdIzL!$1L-L2{??$KOd&nhYPd#P&OI-POXhh6qzQGXcfAMV(P#k5wv!yd`WUaGQNx~ys~d5=riM?n4&Q"
    "TIyjXsuQ6ut##Tm#XZ1)w|x0U4Pm1f7wuWx2mRkXFr<j>sckGez$b%u%x^`3MU^G^+y%*QP+Lc^s=)b%~{@aEw7d5J_=_aJ?~qI>"
    "{~_MtGY|Bt-it@&Gq%Hl2W~ws@AR38G3!2L%vPa-"
    ")45nE!9`}gQ&X7uJ5I)b?bEIUmpYL<L(Pz@c8eAFF5(VuxwpD7Ufx*a?9n|8rfW}C|li+t|EOq^VFg!C$(vX=nU|Ck!W5dv={Z|M"
    "TIH+(xiJSXL{QTAzZCkn$WAeR^79R#UARR9eb%{U;kdL`WM&z#ce$n<yl&HSG_F9_F}=lSTubpdUf~h_|&43O)EraoZm|%{Su+Qq"
    "%Xqrl13GG)lW*>j7w;TQuOvNTKBZ&q&BTmy_0^oD(+UH-"
    "P#x7xwTP;yXs{*wp%+}aiP^+xQkXjZK>+g7ttB@_fpZkv}{GKOY<zPyQ_Xuj;#>Q)ylrechS11C?~aPg%F(dyG=B=RlTUkQi|Iuc"
    "?7prq$|hPo7)7_7x^w)^|a-"
    "r9@?>&3jE#P%iGI#dsQlNd)X>=yU^R#nv&jLwyMZ?(T<N;y06`LpQ^tN{8qj4`FZK{^78lNn@`2}z5nF<V&w0*FAe%2d;gVn^*id"
    "y7t@)~pNn5ISH4w#NBdQB^_%1FSH#;-gg0LTH@^4Xe$QL@c6awt?&^Em-nX^euV{;($-Yzm8n*O>YvGI5;%BQ%BY%*({mS%Fb{~-"
    "Lp8iYF!bhLMFF6a}XYRkoEPiZR{*-"
    "e2)nx61$l6zrwXYf1Un*`tQ7nIiSoqd3@F8LG!@&LLf3;8hYM<`aKF{lYaM%8fuK9Uf>7%#G=WXF{)e7IA?LRateMnaLWNiOESo`"
    "a*?blq-%I<@$2ibg<Rs8xY_c7J(!>HoNPld<&imCLCQu!OC%_m3AuZXJO2kk!r3V-"
    "19zI)|wb>1g`kMln9Yn$yyHO=p3wzB>1rS@sd<^z`HCo84zQVQgL_3M)N-=}^oQvD>P`l(0#8;-"
    "S*zQ!njcd;wNrxX3JBfh+s>Q@bw4;C69By7Jv*vj@pg54(r53>0hp!&r>^*es@xBAKt^YvfhYrm_fe@RdNP+l#HPu;y-"
    "|Eyi}89U`$b;if&ULl$CYjWau<Fv2B)n9#+zw%a#?t5+Wr`h^1uZf>mYrrejucp0H`W9OAxwGcuX3Zze>`#>SMf%oQ|2;A7dtvHV"
    "z{HPzm7n%%Kij2#rAzq;m;9+M<r7-kXR_+gU&-ISa=vG!e56YJ2vz&Fss6)K@~5MU&q2M)`TA4!MW<IO-"
    ")MT3{*|WsOH4}@zOf{KNU8cHlK5Su_RB}&CyvDL7%d+iGCoYye{kqEua-"
    "V7Bz{7u`bv=WRUq+mKi1cLq;K^|ALVK77t5d6DZi%Eele&2JkHXs&)oFiu#vu6(|(tx|Ne~jnVHx2TKZay{5_bZi=TQCzU0z;v!("
    "qeOaBFy=9?;CY<)ka|42&vL6qk6C(ZXx+Ao{bLi|EW`B9Sovm@^JMTBpL2%iP%z5t?s-"
    "b4C`hxT<2?E@Uemo&Zv`+COfU)OvZ<Mpo-"
    "K7i4F@WT9*Mf<&q_A?Z(6F)oAd`;qY&gUbfZ$y~ie$YPZ(0sf>{Xj$iAqME<3iYQG`tKyXLAbS#9VnkMP(D;3e~du-"
    "_<;0Hf&QBT{bvEhcmFr<`fuOuFTK9Md{MvtPX7Kq`{nEP+ZXAVFVA1P_CEafo%i+2?VC5*w=b?oUQOS>YhHhuy!@*8=9TdBTi?s~"
    "yDzV57hmLFy{7$5Z+xTk0`@nxuURie{qFSUW$ERs(S;YFn=d(k^PBp&m%sT<^2^DEw~+U58t>mIUcNqDc|&-"
    "kze0HV_gB1Gc%`@S>hAu{+}(S)`?qZ`uh|w}rER}6yMGOK^V;kB4cGF^teY2BYwx7CUN9}aOIm(mwD3M?>HW|BJD$atIJd86{>uJ"
    "U%m@2b?Ms)-mn$#5e|vIOs=Imn@$tRJZ}Hz_yu7gZ@S5W0y~Oovh?SQOcP|v~Um9$`8u)9<-"
    "oEO$eVwoL>fWVwZ{uBF!Fza{?pGu4)LmYh+r1ulc^|Iy`rGyux7BNGzeaptZDGDncpYu~%GqOE-"
    "Yna`KelZ3+hLbi!Pc*VRe#-UT*WuF7H?v$Ucg$vbhUlO>fw#4)r(WVu6;LZ{XSIv-KWyaO{JHZ9$NXDQs+IS<$Fe@cZx23d2ML@p"
    "3v@HpyJy;7x#_Q`#js1b@p%NJc#<`o81dGm$zybFUowY`j*V)jhOZ8FPk@BN^iE5-e1|ip0a-"
    "}<=bi{yk4??fu!`R$YVdW<yDaGYaW}|I96|MJie1rd>Nzk;>GsWip?7p+qWhPZ%2Gb(WSQ?if=VMzN7H%?n?^W*ASMk9IRh4=)Xr"
    "0e-|MAN<Z!0eBPV)?Dy!Q@5s}u|3*9H4R!A_nf6vW<V|tNJKy|wx#{m~i@j-0dUqP~J~Z?#XV_cHkoS+p-XNyE1*~_-"
    "+q~pAbXjlU(%-S=zb(st3)Xv#cU;+TukzkVg}qw}d21B<?kDX{PQCNr%p^JMol4T%lkm47A#XL(-"
    "&cgbc?fxrko2yg$XkEBclY4$;j!MU)4xf_dOJ?^9XH=>yt77oBMtI)8RVTY`a58-"
    "cf06sZt>p8g1%eDes2o%?i2JKCfd76^f!;7?-"
    "kMB4x+yUL}TCLyvxITBM1714egy7_Iok>_g#F@Ao{y0q_<IE@08Hq716%i;afs)dGO!S;J<-Ea`KxLu=gT-h<K~Phn4phSZ^n=-"
    "Zg-"
    "|LEyuow*j;Y{qObV|Jq*uFY3|%ivA<yfA{YF_w2}jrEdK<<{zQ{+wqSoPW$h?vHzaC|L?T*{~Fu>@2S!MUb_6RqNV>D+WK#sz5iZ"
    "0_Fo=*{}r+SUj|DK{V#bX_A&Fn!Tp%nm-;c$eN0g$`B<ZBZC&3k>Tfsdw>R$FE3VgC)mp#4Lv(dt>UW6l?9-"
    "}4?R%@O>#vCVuNYN-^t$v`+t+sry6)fEDfwxY)LU&`-"
    "zDhp>fCo#N`9o;_wQ2ta(!HJAFuq!D^jbE7iw{b+V@uL*LMs0yNxP8dR@BYr0cJW`mc($R?E|>LtT2Q%IhrsgqVGzQJ*N>ClvJwM"
    "U~_ejrxSZzDG#Er%~TixbLZy{77}_t=6ya74-LtRez+q^ip}(_jRi5->0bW>r`Er-fHdoepM~CR`sLTT9w!1^g%IsQ0z;s)$-"
    "`IR=qoWJ*4=$FI9h}TC3h2u7?%(uvlyLaHs0J^oJFH_IgBgb$_H&%hM{Ux4OQ5Ky+pQ0a5)x!G55rOD~mm{k2a0HL=!Ds}6PPt?s"
    "YCF6h55*8XW#*-"
    "K?#|3jnXT0gBSd#SwZe{57+<wvTuYRTW|lE2ZYrPgYBq`LH0x7QB}`UjgTKc(us<j!6{B)T6GYyGsU?4{DLzuBq3+0_0iRo`2^Tt"
    "6(jA8u-`ez;SITC3jwttR_hRc~wkw5sf_Zm+-HX}K;>tIA#~?fPV=K6$L4Y}_Xsb*M}Kr09P{bY=gMN-a;R`u-!G|IsG-QKA26rI"
    "tskOZTq7Q}wpiPpSIuU4OUgZLOcWF1=LJ_4gXp*8ZvM(k1uz&igUZ{n)YA>g?01vj3Rif4oY5T<AYusk4t%mu_AEQ=^tDKXt8@eE"
    "meHe&VQqqI>GPbjkgF(f$3V)=#OHUMlJO2aQ^4ss2Hu4z*U&^*=Xisq%BEOD|P@eX3KRI@VHss!@kptN!}QPW|Lj|77))YU!owuY"
    "cI6rS?x<mu_8;R(+}UQ`e>2*JD+GOnnY@>7^>Kf7Gdebgcbzs7o(Zef?CYe(I=ys(R|WbnE)*s{eGOe!5X*Z}m@~_cMb2nPZiopA"
    "ouz{jAV`wopIYsIs^2=PLelVqKoPE?vKVzUn{UsGo1t&o`>SexXsn(5YW&>eBV=7pwk@;#1dO?DXT;FA4gWj{295PpOvP>R&qVmj"
    "(UHNBzsybEr!%)#3WbonC7F)b$^C`tj?Zbb6`%^G}4{UjIv@mbyH3UAlbz)2jc|?y2k2`|E$L`cjvtu1oK)e^&LSE>B(mS*IVr{<"
    "lsqb$RMqtMdBatG?9wIn-L!*RM3{SH$O6gkE32D)jz2)LNC-uQlq|#OF|JRbKyG==D?A<$B!zdPRP{=)YblscP51sQSO?o<prwdH"
    "o+%U+VH4>eBn`H>$qW-#Dpf)zTODFDvpdi{3wnTC2nLn~nNS@%hc7FYdQ0@>@mkpWiC_;(og#zg_hDsjGJVPSJm-"
    "P<35;dHrsqez$t+TC3kZ@Ao?WdxiSFqyN34%h&(e)U!`rm%d!T-"
    ">BcOp1S^irT<mI{#8@YK6TZue_izYsjGDTo2J&!zbU$O{XtWg=TK{P_WHv{{bBVS>eAcm-xj@p{%z6w>;G!%@*HZd9>4xwqyAm@9"
    "IAHx`=<W;lX_Myy}tga=>79YMQ^YFyXf_EsI_|h`VW=*4~43$eEmN~U!J<Q*MF?~e>^^Qm9GD{sb`-"
    "<UHWqUan=90dJeT#e|+A5YSe!^K6RC^|L<6r=TMh^xc+n1|8w;mYOVfDqy9_v9O}}~UjMaH|Fuwuy7cAxldAv8@j295J%0UZ)&I1"
    "5{<P@r^%tLXsP*&sb?D{w&Z5`PJBwak?<)H8)U~`GsQLrP=THyVyPov!!o9ny?>^D*KDie4o<_Z=dfrp?@_KL4m*-"
    "HI{`mF2s=u##4z<1BU-acURO$NWV_lwaE_!)=py<o<fuh&fUpm(1IaKNTmSa8p9BQdPcqAV@*0ax{F6u)^e^!6!|8F}kfaBP*Gmh"
    "8T()2V9iOu+!-&<zhY&M&0vSFqQRFX<6mDE<CBCsTj$zo<kOR|`mEQ@84g%+~~wk+lJ>DzaDw6A{k|IWSS6==2R1<|#n?0La-Eh+"
    "n}Lg>mVYq=p8Lakobg{rPB%<6_=%37}Efz`{Bvgd`<wWO@I2)dS(wH9f$kyd+N6kR!G&x^L&XscC=vDz5&j`>`%bYsu0S{z+V%Bs"
    "a%ZM@Z5OQ0*K?0JcF<q}obB8jdgnfywo|75z!bS*c9T=`{5S+!KUmQqzW^+KA}O~aHuFP*NOvT7N0Eq$I$x^gn*m1T9aFlEonrkh"
    "Royd1i6IjSqkRb4q{t>w|pqiebORyQBZS6xK`T{&gd3hBxfs;;C+b>);*E2dja*K$i#*V5;CXw~N_wdz-"
    "u(JiBEt(<JR)hZrY?IW_*D(F_wRjZ_1N!MBx-"
    "72~k)pX^QRn*XxQ&v{1x^lIuTYI5Sb?fM!S5H&cqJeG$UA0EKjdU%V=*s=7W}32!7P@lE%3A5lDJyHEYe}u0ZaZCz4!Rw56`geDW"
    "E~f}=(~%itgM@^oU)Q0)$O5c(Mz|NuA+}_A6>;`x{v8v^pou;YwZb5S=j(xIctMt2g#}pksTtdHcV4iGNQU8bQPmyN6E@}jJ(HaW"
    "Au)b-"
    "#DhM;wjyyNWK%~H$gjZl4fm+>=apxX{#(|=*}SdJ|n+pn6ipltJLOb%1Y+x%Bd~Tl$9*fU8E~pBD+M^;yKypWEIOaWhE<gSLjMs>"
    "8>LAu94pwcHTNoZG-FvS=lDpO|lkSG_`G-vfCGS=<d*!?9$ymXOCuWA5&KHg6@lR4rpqJG_@m|+A&S-"
    "B~9%WP3?qc@tW*wd`WvsS59$8_Kd8=^`f$_7uDQoYVMe_5)Z3n_sHHOEAu4li7#or=z5*wjaj>oDeHqPT_3VCUz&v<S!@27Ke++;"
    "3c!>Nq#I}@h^7dpsfA!_p)^?-*)Xz_2V@`ME1bN-F?l2KCEo~@NSY#wrii9l#L#51WMj$3UW}tj8;>^*Z@iTROf3;pO9JxBl4-"
    "IOnnfz6Y#P}#vS}C7$)@8ggS;~El|f|E6j_+E*<`cHW?#&qS>$4Bc|cxCKBjB|*#fc!7Yk{MBAToiQ?`U`3E78aACfJlTS}DC6y="
    "!3Bg~=#v#7)@sxU=0O;STs)M6HOKwe2bX3>CIG-8S-"
    "Owmk}w9sU&n6zzpTj8p;W3moR+D^P3a69qcg|`cnbd$B{!DPKONgqw}m?r6`$(~?}0nB0$4O|?;JA`-"
    "W;xN93@eW@c!PkhDQI#<u?>Je-Q%o^|$tE$2DIo1MzNS@XFv&C8vx~E26?2$k9+NE47A`K5RV-"
    "n$=a^y{NV|fs6_r&W?HWz8PTPQ_wb)eI0@7~NBs-X57nAH^ihWG-0#h7dvO|?4Omd7VUaGuOIiX%-"
    "l2c4^2BdYhk&@QUM!Ig4yNU;O50iLeGA~TxjVbO^KA4Y<FU=2<R_2dM0x(%1CJDkM!I&b12*qS!K-"
    "veGBAkk#BC$xDC`=Je#9)$GOcqDQV+oM7l0;0FgeBP|W3m)16_Qq#Mx|rvHW`>C6O&|Nl59+p1EkHxWO-CRRzNC*Er693Va2o(Oz"
    "{wvVv;hf%%&WZJt8Ww3Y$tyQiWC7RAZ7Ftj4AmlhqORR0GjSG+|9P&6uQxY9-"
    "o;cC3Tc2`kryNxHFan;xvkrWcd+QICm!;t4T84Wc1x7>!^fHlx_6%@{UjGmeehJjEmv#3V6=O=HtGGnnKVF-"
    "y!5^Uwn90<lOeq375#CRw3Y!5X$^vrcWGO==5lgDsmK*d4Ts?O}U1``8OiazGpsN9dS%iCz&W#A|d)oFP|RDK}dwcTD1e?qQyo7b"
    "fv0?gJmp7nArQe<A<`Vu7|nL@){=LQxocK!k$`EE0>fjUu8^3=s?Bh<KEM5{V>`jHO_ySQ?gQn@(haOe~AYMmbO}Y_4q{mS>w!6o"
    "5ja2oysluqEgrD22*k%TPIZgjEogpbDyntwuGV7S%!Zu=St;HKHcej9Ne|)CSuI+EEA83EOGgMRbE6s28>u^r6R~pLhZW&>$FshG"
    "B=n2pR=r#JKHKXaaTuOrj|?4Q9|YGz;d?JXnAhVHa(ei08J;#0ppiYtTCEy6pzBX}d*i+wMTSu)AOn?4uXp03D(u+hgLT?JIPGUV"
    "~F`2DxfwZd!NZp4J1pr}YG0$Q#@TKFAmN0e=(#0#T4Q7=@rvZ5Vn0!chc>)J9>^AO?y>aVTD!fD%CxNCqh=6{LZ5kO4ALmNpy8LA"
    "fYjn+Fx3LTwRLj7q>mRHiM3%F!cj1*+6mp=wYAYC#>S2MwSRG=XN&g4(pLP&?=Vov2&e1@(Yl&<7rae((egfI%<>hP5MT6pU%d!B"
    "aG$odi>08q9!a+F3NGokt7WMX&^(YnRcAb``B@*TIH%6K!d?!4B93dthJt0vv!taHKs(FSW0<C(vtfid-3q8{-"
    "Z<z&*wjc{5(jedNpdFn-"
    "9N2>^ji5DH;}nNSc09)NHr0z`snCW?uHjRkQao=E_SAc;u^DNHI#2N@s}WHH%H4$5P4nS4;d6ry5K0v<A@pbR`>%9#pK396YYrUu"
    "l4I;I{pFpZ#zX=Ykb8)yd|Oeg4Kx|tr(3m!9lOh0(S41hsq2#hep%qSRR#=%o&0!%Vf%ruw*vtSO)GYiZjSYn<t%V33B1#8SYvjH"
    "}lEoK|+FuPzM9e_jT2plsnnOER7Is>jcH{b#80Z*Mb@&Ud&Kj5zm)CK5*bip7*7YZJLa9sq50?{BA#G^!zj8b)JAYGTC%LLgVN0$"
    "roK_MstCE%g1OjoKa*F6H2ph{N_YISv>Ue};&)HQ(?&<5Id9iR*JfL>jn?y;_4_e3`U26e+=L^rA%(~aw%f=Mt9W^~VVv${Fmylz"
    "3a2%du#u&P@J8@f&1mTp_OqubT(>Gr{a?hqV<SKu``)w!~+tQ+gj-UFVj7wgUX0AJRR^=AXvKsFeJu%T=i`+yB+BS91!&Bm~CAf8"
    "PG$smPIW79z<n+0;%TsDu*XA9UuwumieOW0CS4l3A6P|en`wQL<*&o+W)(8{*4?Q93z$#${bY%h4s_Onme0d|lbVn@IjJI+33C)i"
    "1L8a!iX**SKeU0|2MGP}aAfpvD1-D0=d9d?i12M6pCc*(wEPuSP&DeKC)aqgT4=gE0--rRl87x;4lTo4z`g>a!<7#Ge(aFJXzhy("
    "FlBA3LaaH(7xm(FEySzHdt<MO!zu81q<O1M(4j4S6VxJs^ytKn+7I<A3h<eIn^u9a)&I=D`*i|gU~KtDIY4RS-"
    "=FgMDLb5FSmZjzhko^i9>9JjzNa!cGYx5}+?>)a-{#qDsr+#dIWJLHbIW9}7q!kuzwoEz@}ynqky%lq*Gd>|jphwx#1I3K}B^3i-"
    "OAIB%~iF`7j!l&^Wd?ug8=Yj&hkT2pNf^z;5U%^-NHGC~!&o}Z-"
    "d<);kxAUERH{Zke@%{W0evlvLNBA**oS)>U_!<5gKgTcdOZ;<wg<s=0_)UJB-"
    "{trC1OAXd;$QJ6{26c;?g^fPx8NiA3I0N$5F&&MVM2ruDZ~h|LcEYDBnv4*x{x7c2{}TZP#_cuB|@oCCR7NOLbXsQ)C-"
    "M5v(PHE3!Or@&?`I^o(O}&kT5EY3lqYWFeA(g^TML=Tv!p-gbiU^*b(-G1L073DVzwWf~)8*-V?n<AJJC~5QD@JF-"
    "(jQBgGgoPD~J!#8fd|%n)<LT(Lkb6idW1v0SVatHnC8UThLu#CEY$>=t{)esMq?7DvT#aYCFHXT>>jQCt>R#0_y%+!6Q11Mx_FCB"
    "7D2_3nC4y|>;+AD|D^hv*;ZBlOYwSbc&%NuR3E&}Zp$_4)cDeTlwIU!kwk*XkSeP5M@ShrUbStMAtj=!f-"
    "V`U(A%{+WJWzocK*ujx1SJNiBSq5fEZqCeBS89WW%20uf9A=nURh%iJM;tYv~6hoRJ%aCIzFccX|4dsR^L#?6S&}?WkbQ-"
    "!1{e}U<uwl$FX_z+585Rx8hIPZHVb}1&aAY_!oEqGXUdH=Ie`Am_%ouKrF~%8_jH$*<V~#Q3SZpjcRv4>|^~NS+tFhDAYwR};8Ap"
    "v1#%bf6aml!1+%Rq%_l-"
    "x!6XTiD!{lZ1H3gYMOcADNQ@knJly1s4<(rC3rKSo~jj6%ZV(KvUnEFjarZLl`>6vN4v}{^8ZJS=0j!dsjZe~xjk2%mBYK}0+m=n"
    "yY=1g;*xzJo@t}xe{o6N1|ZgZb`$UJJEG|!rs%&X>2^Pc(0d}4O9^Rn}^3$lA)7iE`VmtvP`muFXOS8i8j*I?IT*JamdH)uC*H)S"
    "_(_uOv7ZrAS6?!?aB-pk(KKEyuKKGr_jKGQzWzSzFpzQ(@EzTLjZe$al@e#(B{e#L&%e&7Ds-qpd=!OtPYA;KZvA;lrfp}?Wcp~|"
    "7bq0OP!VZdSBVa8$6Va;LJ;mG0C!PC*tF~l*-F~Kq2G1sxgvC^^LvCXl^anSLp<E-"
    "Pd<EG<*<B6k(lb=(FQ<PJZQ>IhCQ<+n(Q?paI(}2@cr&*^Jr){T0r!yyS=OE_@=LF{r=X~dK=UV4h=RW6Q=PBnU=S}AW=Tm1dmmr"
    "rYmqeE=mtvPnmnN5PmqC|FmqnLNmqV8`7oSU^mtrraUCO`o=u-Wqu1f=#rY<dB+P-"
    "vr$>XyB<;crPmvb(cUar5~ae46a)aB=wcQ2n@_PP>!CH6|@mEtQkSK6-"
    "(T$#GEd}aU2=@s9r;a8Kd=3jkuwdHF6)rqUmukKwvyXt=}@><%p;%l|nI<Jjfo4>Yw?c|#8^@!_f*Nd;$U+=j-"
    "etr4+i|g(;LT)79$iGp2qw~htjinn0H#}~J-AuVze6!)^<C`-#H*cQa47`<aEB{vAt-f0`x3+Gb-"
    "3qy#e7od!%kAOYOSg}1``(GYQ*fu@&cL08JBN3C@5bLPy4!qr^zQ22)4L&WWV})J#^X2U-"
    "#B_B;LYSWE8gsVbMDRKHv`{Ff2;bffwxxPa(g@S?ZUU)-"
    "=2B<@a>>?vfgQW=jl89?*zP?`EJv@6Yn0r8}eTMdmZmByyyCU{QH&fkG#M4e#i#}AM}2(_JQw**&lX(xcs5dN7)~BeYEybz{iCjK"
    "lyn7<H%2{Kbii-"
    "<I}89dq3U#H2SlK&z3$5`a<~^X1?J4#gZ>he9`Mm#b28GlFygRzdZltkgqg+W$P=6U+w>@>(?HBZSiYSU+?+4+c(O;vGI-"
    "6Z%%yk!MA$9b^qH<-"
    "#+_x&3BH!Q~BMa?^b^A<@f5o@BV}K9|Zhx=!fw?TKZARk57Nx{cEv5S^G)NuLu8h@uyY45%#n7pSAyH+HZOMeE#Qczn$|t!M`~A#"
    "r*H~|6bki7yd!&AIAJq$R7v%iQk|4|5@;#NBl+7U*`W+-"
    "Cqy<&C1{2`@7V?Z~uqQe~kI3&VN4pm&$)V{I{Ba_xz9H|IGcR*MF`2_rU))|6jG6rrJZ(?5P>@)@=D`!u&OjfturBO<kBKFha8$r"
    "5TRZ^dxA8k~OPon$RpwPp&4qQ1hxpvs12dt=44KYgU^zogJFaUd{4=CVxy5Gp#9F(8NV*j(*wyKkcrX*?XEqAI<$hO?;SUIZCshp"
    "eae$H05bBN;MlbnzS~}bibx}Qd7O6Nj%c{`dp}vyWslpLTm4Z)Qt-VVHZ7GF7~?F<PF*+wc3_#X`6l7l~yq>_R9YQJ4N3+"
)


def _decode_positive_quantile_bits() -> tuple[int, ...]:
    """解码正半轴递增 float32 位模式并拒绝任何表内容漂移."""

    compressed = base64.b85decode(
        _POSITIVE_DELTA_VARINT_ZLIB_BASE85.encode("ascii")
    )
    payload = zlib.decompress(compressed)
    if hashlib.sha256(payload).hexdigest() != _POSITIVE_DELTA_VARINT_SHA256:
        raise RuntimeError("标准正态分位数差分流摘要发生漂移")

    values: list[int] = []
    previous = 0
    current = 0
    shift = 0
    for byte in payload:
        current |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 35:
                raise RuntimeError("标准正态分位数差分流包含越界 varint")
            continue
        bit_pattern = current if not values else previous + current
        if bit_pattern > 0x7FFFFFFF:
            raise RuntimeError("标准正态正分位数位模式越界")
        if values and bit_pattern <= previous:
            raise RuntimeError("标准正态正分位数表必须严格递增")
        values.append(bit_pattern)
        previous = bit_pattern
        current = 0
        shift = 0
    if shift != 0 or current != 0:
        raise RuntimeError("标准正态分位数差分流以不完整 varint 结束")
    if len(values) != _POSITIVE_QUANTILE_COUNT:
        raise RuntimeError(
            f"标准正态正分位数数量不等于{_POSITIVE_QUANTILE_COUNT}"
        )
    return tuple(values)


@lru_cache(maxsize=1)
def standard_normal_quantile_float32_table() -> tuple[float, ...]:
    """返回按20位索引排列的规范量化标准正态 float32 值.

    该表属于通用的跨平台逆 CDF 实现结构.其他项目可以复用相同的索引规则
    与表摘要,在不依赖系统数学库的情况下生成逐字节一致的正态 Tensor.
    """

    if (
        struct.calcsize("f") != 4
        or struct.pack(">f", 1.0) != b"\x3f\x80\x00\x00"
    ):
        raise RuntimeError("宿主平台没有提供规范 IEEE-754 binary32")
    positive_bits = _decode_positive_quantile_bits()
    negative_bits = tuple(
        bit_pattern | 0x80000000
        for bit_pattern in reversed(positive_bits)
    )
    full_bits = (*negative_bits, *positive_bits)
    raw_bytes = b"".join(
        struct.pack(">I", bit_pattern) for bit_pattern in full_bits
    )
    if hashlib.sha256(raw_bytes).hexdigest() != NORMAL_QUANTILE_TABLE_SHA256:
        raise RuntimeError("标准正态分位数完整表摘要发生漂移")
    values = struct.unpack(f">{NORMAL_QUANTILE_COUNT}f", raw_bytes)
    repacked_chunks = []
    for offset in range(0, NORMAL_QUANTILE_COUNT, 4096):
        chunk = values[offset : offset + 4096]
        repacked_chunks.append(
            struct.pack(f">{len(chunk)}f", *chunk)
        )
    repacked_bytes = b"".join(repacked_chunks)
    if hashlib.sha256(repacked_bytes).hexdigest() != (
        NORMAL_QUANTILE_TABLE_SHA256
    ):
        raise RuntimeError("宿主 binary32 解码没有逐字节重建标准正态表")
    return values


def standard_normal_quantile_table_record() -> dict[str, object]:
    """返回只由采样数学定义和冻结表字节决定的算法身份."""

    return {
        "normal_quantile_table_version": NORMAL_QUANTILE_TABLE_VERSION,
        "normal_quantile_table_sha256": NORMAL_QUANTILE_TABLE_SHA256,
        "normal_quantile_index_bits": NORMAL_QUANTILE_INDEX_BITS,
        "normal_quantile_count": NORMAL_QUANTILE_COUNT,
        "normal_quantile_probability_rule": "(index+0.5)/1048576",
        "normal_quantile_value_rule": (
            "round_binary32(inverse_standard_normal_cdf(probability))"
        ),
        "normal_quantile_symmetry_rule": (
            "negative_index_sets_exact_float32_sign_bit"
        ),
        "normal_quantile_maximum_cdf_cell_width": (
            NORMAL_QUANTILE_MAXIMUM_CDF_CELL_WIDTH
        ),
        "normal_quantile_ideal_midpoint_ks_distance": (
            NORMAL_QUANTILE_IDEAL_MIDPOINT_KS_DISTANCE
        ),
        "normal_quantile_float32_cdf_rounding_error_bound": (
            NORMAL_QUANTILE_FLOAT32_CDF_ROUNDING_ERROR_BOUND
        ),
        "normal_quantile_float32_ks_distance_bound": (
            NORMAL_QUANTILE_FLOAT32_KS_DISTANCE_BOUND
        ),
    }


def standard_normal_quantile_reference_record() -> dict[str, object]:
    """返回独立 MPFR 正确舍入复验身份,不改变采样算法摘要."""

    return {
        "normal_quantile_reference_verification_protocol": (
            NORMAL_QUANTILE_REFERENCE_VERIFICATION_PROTOCOL
        ),
        "normal_quantile_reference_precision_bits": (
            NORMAL_QUANTILE_REFERENCE_PRECISION_BITS
        ),
        "normal_quantile_reference_newton_iterations": (
            NORMAL_QUANTILE_REFERENCE_NEWTON_ITERATIONS
        ),
        "normal_quantile_reference_verified_positive_entry_count": (
            _POSITIVE_QUANTILE_COUNT
        ),
        "normal_quantile_reference_mismatch_count": 0,
        "normal_quantile_table_sha256": NORMAL_QUANTILE_TABLE_SHA256,
        "normal_quantile_reference_mpfr_rounding_mode": (
            NORMAL_QUANTILE_REFERENCE_MPFR_ROUNDING_MODE
        ),
        "normal_quantile_reference_verification_digest": (
            NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST
        ),
    }


__all__ = [
    "NORMAL_QUANTILE_COUNT",
    "NORMAL_QUANTILE_FLOAT32_CDF_ROUNDING_ERROR_BOUND",
    "NORMAL_QUANTILE_FLOAT32_KS_DISTANCE_BOUND",
    "NORMAL_QUANTILE_IDEAL_MIDPOINT_KS_DISTANCE",
    "NORMAL_QUANTILE_INDEX_BITS",
    "NORMAL_QUANTILE_MAXIMUM_CDF_CELL_WIDTH",
    "NORMAL_QUANTILE_REFERENCE_MPFR_ROUNDING_MODE",
    "NORMAL_QUANTILE_REFERENCE_NEWTON_ITERATIONS",
    "NORMAL_QUANTILE_REFERENCE_PRECISION_BITS",
    "NORMAL_QUANTILE_REFERENCE_VERIFICATION_PROTOCOL",
    "NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST",
    "NORMAL_QUANTILE_TABLE_SHA256",
    "NORMAL_QUANTILE_TABLE_VERSION",
    "standard_normal_quantile_float32_table",
    "standard_normal_quantile_reference_record",
    "standard_normal_quantile_table_record",
]
