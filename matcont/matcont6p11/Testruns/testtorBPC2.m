p=[0.5;-0.6;0.6;0.32858;0.93358;-0.9;0.001];
x=[0.00125;-0.001;0.00052502];
[x0,v0]=init_EP_EP(@torBPC,x,p,[6]);
opt=contset;
opt=contset(opt,'Singularities',1);
opt=contset(opt,'MaxNumPoints',10);
[x,v,s,h,f]=cont(@equilibrium,x0,[],opt);
x1=x(1:3,s(2).index);
p(6)=x(end,s(2).index);
[x0,v0]=init_H_LC(@torBPC,x1,p,[6],0.0001,25,4);
opt=contset;
opt=contset(opt,'Singularities',1);
opt=contset(opt,'Multipliers',1);
opt=contset(opt,'MaxNumPoints',50);
[x,v,s,h,f]=cont(@limitcycle,x0,v0,opt);
[x1,v1]=init_NS_NS(@torBPC,x,s(3),[6 7],25,4);
opt=contset;
opt=contset(opt,'VarTolerance',1e-4);
opt=contset(opt,'FunTolerance',1e-4);
opt=contset(opt,'Userfunctions',1);
UserInfo.name='epsilon0';
UserInfo.state=1;
UserInfo.label='E0';
opt=contset(opt,'UserfunctionsInfo',UserInfo);
opt=contset(opt,'Backward',1);
opt=contset(opt,'MaxNumPoints',16);
[xns1,vns1,sns1,hns1,fns1]=cont(@neimarksacker,x1,v1,opt);
plotcycle(xns1,vns1,sns1,[1 2]);
