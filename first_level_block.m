clear
 
clc
 
%drive='g:\fMRI-MathProbSolving\'; 
drive='g:\fmri_arithrhythm\'; 
 
spm('Defaults','fMRI'); 
namelist={'addlarge','addsmall','multlarge','multsmall'}; 
runid=2; 
 
for subid=2:2 
    subid
 
     
    %First calculation then reasoning 
    %if rem(subid,2)==1 
    %    sots={[3 93 183 273],[365 455 545 635]}; 
    %else
 
    %    sots={[365 455 545 635],[3 93 183 273]}; 
    %end
 
     
     cch2sots={... 
           [16 61 106],[1 46 91],[151 196 241],[136 181 226],...   %% sub1  加小 加大 乘小 乘大 attention to the comma 
           [1 46 91],[16 61 106],[136 181 226],[151 196 241],...  %% sub2  加大 加小 乘大 乘小 
           [151 196 241],[136 181 226],[16 61 106],[1 46 91],...  %% sub3 乘小 乘大 加小 加大  
           [136 181 226],[151 196 241],[1 46 91],[16 61 106],...  %% sub4  乘大 乘小 加大 加小 
           [16 61 106],[1 46 91],[151 196 241],[136 181 226],...   %% sub5  加小 加大 乘小 乘大 attention to the comma 
           [1 46 91],[16 61 106],[136 181 226],[151 196 241],...  %% sub6  加大 加小 乘大 乘小            
           [151 196 241],[136 181 226],[16 61 106],[1 46 91],...  %% sub7  乘小 乘大 加小 加大                
           [136 181 226],[151 196 241],[1 46 91],[16 61 106],...  %% sub8  乘大 乘小 加大 加小 
           [16 61 106],[1 46 91],[151 196 241],[136 181 226],...   %% sub9  加小 加大 乘小 乘大 attention to the comma 
            [1 46 91],[16 61 106],[136 181 226],[151 196 241],... %% sub10 加大加小 乘大 乘小
            [151 196 241],[136 181 226],[16 61 106],[1 46 91],... %% sub11 乘小 乘大 加小 加大
            [136 181 226],[151 196 241],[1 46 91],[16 61 106],... %% sub12 乘大 乘小 加大 加小
            [16 61 106],[1 46 91],[151 196 241],[136 181 226],... %% sub13 加
            小 加大 乘小 乘大 attention to the comma
            [1 46 91],[16 61 106],[136 181 226],[151 196 241],... %% sub14 加大
            加小 乘大 乘小
            [151 196 241],[136 181 226],[16 61 106],[1 46 91],... %% sub15 乘小
            乘大 加小 加大
            [136 181 226],[151 196 241],[1 46 91],[16 61 106],... %% sub16 乘大
            乘小 加大 加小
            [16 61 106],[1 46 91],[151 196 241],[136 181 226],... %% sub17 加
            小 加大 乘小 乘大 attention to the comma
            [1 46 91],[16 61 106],[136 181 226],[151 196 241],... %% sub18 加大
            加小 乘大 乘小
            [151 196 241],[136 181 226],[16 61 106],[1 46 91],... %% sub19 乘小
            乘大 加小 加大
            [136 181 226],[151 196 241],[1 46 91],[16 61 106]... %% sub20 乘大
            乘小 加大 加小
 };
 
 outpath=sprintf('%sAddMult/FirstLevel/sub%.2d',drive,subid);
 if isdir(outpath);rmdir(outpath,'s');end;mkdir(outpath);
 jobs{1}.stats{1}.fmri_spec.dir = {outpath};
 jobs{1}.stats{1}.fmri_spec.timing.units = 'scans';
 jobs{1}.stats{1}.fmri_spec.timing.RT = 2;
 jobs{1}.stats{1}.fmri_spec.timing.fmri_t = 16;
 jobs{1}.stats{1}.fmri_spec.timing.fmri_t0 = 1;
 
 sots=cch2sots((subid-1)*4+1:subid*4); 
 
 for tid=1:1
 % taskid=find(taskord==tid);
 
fpath=[drive,'Preprocessingdata\sub',dec2base(subid,10,2),'\run',dec2base(runid,
10,1),'\'];
 %inpath=sprintf('%sPreProcessing/sub%.2d/run%d',drive,subid,taskid);%注
意这个地方，比如第三个被试，run1 其实是 task3，所以 tid 是 1，taskid 是 3
 jobs{1}.stats{1}.fmri_spec.sess(tid).scans = 
filename_list(fpath,'swra*img');
 ncon=4;
 for c=1:ncon
 jobs{1}.stats{1}.fmri_spec.sess(tid).cond(c).name=namelist{c};
 jobs{1}.stats{1}.fmri_spec.sess(tid).cond(c).onset=sots{c};
 jobs{1}.stats{1}.fmri_spec.sess(tid).cond(c).duration=15;
 end 
 
 
 end
 spm_jobman('run',jobs)
 clear jobs
 %******************** Estimate ********************%
 jobs{1}.stats{1}.fmri_est.spmmat = {fullfile(outpath,'SPM.mat')};
 spm_jobman('run',jobs)
 clear jobs
 %******************** Contrast ********************%
 jobs{1}.stats{1}.con.spmmat = {fullfile(outpath,'SPM.mat')};
 for i=1:4
 jobs{1}.stats{1}.con.consess{i}.tcon.name=namelist{i};
 con=zeros(1,4);con(1,i)=1;
 jobs{1}.stats{1}.con.consess{i}.tcon.convec=con;
 end
 jobs{1}.stats{1}.con.consess{5}.tcon.name='add-mult';
 jobs{1}.stats{1}.con.consess{5}.tcon.convec=[1 1 -1 -1];
 jobs{1}.stats{1}.con.consess{6}.tcon.name='mult-add';
 jobs{1}.stats{1}.con.consess{6}.tcon.convec=[-1 -1 1 1];
 jobs{1}.stats{1}.con.consess{7}.tcon.name='large-small';
 jobs{1}.stats{1}.con.consess{7}.tcon.convec=[1 -1 1 -1];
 jobs{1}.stats{1}.con.consess{8}.tcon.name='small-large';
 jobs{1}.stats{1}.con.consess{8}.tcon.convec=[-1 1 -1 1];
 jobs{1}.stats{1}.con.consess{9}.tcon.name='add';
 jobs{1}.stats{1}.con.consess{9}.tcon.convec=[1 1 0 0];
 jobs{1}.stats{1}.con.consess{10}.tcon.name='mult';
 jobs{1}.stats{1}.con.consess{10}.tcon.convec=[0 0 1 1];
 
 jobs{1}.stats{1}.con.consess{11}.tcon.name='addmult';
 jobs{1}.stats{1}.con.consess{11}.tcon.convec=[1 1 1 1]; 
 
 spm_jobman('run',jobs)
 clear jobs
end