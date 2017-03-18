pro gd_pmi,PMI_daily,ch_sample
;;Input weather data and generate the PMI index for each day based on
;;the weather data.

;;Right now assume that 6 hours=6*4=24 measurements of temperatures
;;with values between 70 and 85 F

;;This chooses which data table to use.
;ch_sample='Laguna Upper'

;;Read in the data file
case ch_sample of
   'Two Rock': file='two_rock.txt'
   'Frei'    : file='frei.txt'
   'Laguna Upper':file='laguna_upper.txt'
endcase 

;;Read the information in the files
readcol,file,date,time,temp,pp,wetness,RH,skipline=1,format='A,A,D,D,A,D'

;;Initiate the date
start_date=date[0]

;;This array holds the daily temperature properties
;;=0 if count_temp<24 and check_max_temp=0
;;=1 if count_temp>=24 and check_max_temp=0
;;=2 if count_temp>=24 and check_max_temp=1
;;=3 if count_temp<24 and check_max_temp=1
;; day_temp_meter=[]  REMOVED BECAUSE DIFFERENT VERSION OF IDL?
day_temp_meter = [-99]   ; Will need to remove the -99 from the beginning of the array 

;;For each day loop over the measurements and record the number of 15
;;min intervals that contain tempeatures in the sweet spot.
check_max_temp=0      ;;Variable signaling one 15 min interval with temp>95
count_temp=0l         ;;Counter for the number of 15 min intervals with 70<=temp<=85
date_check=1          ;;Check if I am still on the same date

;;Loop over all measurements
for i=0l,n_elements(date)-1,1 do begin

   ;;Check if it's still the same day
   if date[i] ne start_date then begin
      ;;Populate the array day_temp_meter (in the end this should contain
      ;;the exact number of days that were measured)
      if ((count_temp lt 24) and (check_max_temp eq 0)) then day_temp_meter=[day_temp_meter,0] 
      if ((count_temp ge 24) and (check_max_temp eq 0)) then day_temp_meter=[day_temp_meter,1] 
      if ((count_temp ge 24) and (check_max_temp eq 1)) then day_temp_meter=[day_temp_meter,2] 
      if ((count_temp lt 24) and (check_max_temp eq 1)) then day_temp_meter=[day_temp_meter,3] 
      
      ;;Reset the arrays for the new day and assign the new day for the day checker
      check_max_temp=0  
      count_temp=0l       
      start_date=date[i]
   endif

   ;;Check the temperature
   if ((temp[i] ge 70) and (temp[i] le 85)) then begin
      count_temp=count_temp+1
   endif else begin
      if (temp[i] gt 95) then begin
         check_max_temp=1
      endif
   endelse

   ;;Check if we are now on the last day of data
   if i eq n_elements(date)-1 then begin
      ;;Populate the array day_temp_meter (in the end this should contain
      ;;the exact number of days that were measured)
      if ((count_temp lt 24) and (check_max_temp eq 0)) then day_temp_meter=[day_temp_meter,0] 
      if ((count_temp ge 24) and (check_max_temp eq 0)) then day_temp_meter=[day_temp_meter,1] 
      if ((count_temp ge 24) and (check_max_temp eq 1)) then day_temp_meter=[day_temp_meter,2] 
      if ((count_temp lt 24) and (check_max_temp eq 1)) then day_temp_meter=[day_temp_meter,3] 
      
      ;;Reset the arrays for the new day and assign the new day for the day checker
      check_max_temp=0  
      count_temp=0l       
      start_date=date[i]
   endif

endfor
;; Remove initial -99 entry 
day_temp_meter = day_temp_meter[1:*]

;;Now using the day_temp_meter array I can construct the PMI
PMI=0                                           ;; Initiate PMI at 0
PMI_daily=indgen(n_elements(day_temp_meter))*0  ;;Array holds the daily PMI
PMI_trigger=0 ;;This is used to check if the required 3 consecutive day threshold for the index triggering if met 
;;=0 if count_temp<24 and check_max_temp=0
;;=1 if count_temp>=24 and check_max_temp=0
;;=2 if count_temp>=24 and check_max_temp=1
;;=3 if count_temp<24 and check_max_temp=1
for i=0,n_elements(day_temp_meter)-1,1 do begin

   if (PMI_trigger eq 1)  then begin 
      if ((day_temp_meter[i] eq 1) and (PMI lt 90)) then PMI=PMI+20
      if ((day_temp_meter[i] eq 1) and (PMI eq 90)) then PMI=100
      if ((day_temp_meter[i] eq 2) and (PMI lt 100)) then PMI=PMI+10
      if ((day_temp_meter[i] eq 3) and (PMI gt 0)) then PMI=PMI-10
      if ((day_temp_meter[i] eq 0) and (PMI gt 0)) then PMI=PMI-10
      PMI_daily[i]=PMI
   endif
   
   if (PMI_trigger eq 0) then begin
      case PMI OF
         0: begin 
            if (day_temp_meter[i] eq 1) then PMI=PMI+20
            PMI_daily[i]=PMI
         end
         20: begin 
            if (day_temp_meter[i] eq 1) then PMI=PMI+20
            if (day_temp_meter[i] eq 0) then PMI=0
            PMI_daily[i]=PMI
         end
         40: begin
            if (day_temp_meter[i] eq 1) then begin
               PMI=PMI+20
               PMI_trigger=1
            endif
            if (day_temp_meter[i] eq 0) then PMI=0
            PMI_daily[i]=PMI
         end 
      endcase 
   endif
   
endfor

;;This section defines the PMI data available from Scott
sample_period=indgen(n_elements(PMI_daily))
case ch_sample of
   'Two Rock': begin
  ;;corresponding days: 11 18 26 02 05 09 12 16 22 29 02 09
      sample_intervals=[11,18,26,33,36,40,43,47,51,55,58,65]
      sample_PMI_intervals=[20,0,0,0,40,0,0,0,0,20,20]
   end
   'Frei': begin
  ;;corresponding days: 11 18 26 02 05 09 12 16 22 29 02 09 17 
      sample_intervals=[11,18,26,33,36,40,43,47,51,55,58,65,73]
      sample_PMI_intervals=[100,100,80,40,70,70,50,100,70,30,20,20]
   end
   'Laguna Upper': begin
  ;;corresponding days: 13 20 27 03 10 17 24 29 06 08 15 22 29 
      sample_intervals=[13,20,27,34,41,48,55,60,67,69,76,83,90]
      sample_PMI_intervals=[0,0,0,20,20,60,60,10,0,70,100,100]
   end
endcase

n_int=n_elements(sample_intervals) ;;Variable used to keep the number of intervals provided
sample_period=indgen(n_elements(PMI_daily))+1 ;;The days that have data (assumed sequential but may be missing days, need to have that checked)
sample_PMI=indgen(n_elements(PMI_daily))*0    ;;This is the provided sample PMI
;;Record the PMI according to the revelant intervals and values
for i=0,n_int-2,1 do begin
   if i eq n_int-2 then begin
      sample_PMI[sample_intervals[i]-1:sample_intervals[i+1]-1]=sample_PMI_intervals[i]
   endif else begin
      sample_PMI[sample_intervals[i]-1:sample_intervals[i+1]-2]=sample_PMI_intervals[i]
   endelse
endfor

;;;This is all plotting routines
;;This plots our calculated PMI 
;tt=plot(sample_period,PMI_daily,sym='*',name='Our PMI')
;;This overplots the provided PMI
;tt1=plot(sample_period[sample_intervals[0]-1:sample_intervals[n_int-1]-1],sample_PMI[sample_intervals[0]-1:sample_intervals[n_int-1]-1],sym='+',color='red',/overplot,/current,name='Provided PMI')
;tt.xtitle='Day record'
;tt.ytitle='PMI'
;tt.title=ch_sample+' location'
;leg = LEGEND(TARGET=[tt,tt1], POSITION=[0.2,0.8],/normal, /AUTO_TEXT_COLOR)
end


pro gd_pmi_2,day_properties,sample_properties
;;Input weather data and generate the variables that Nick needs for
;;the machine code.

;;This chooses which data table to use.
ch_sample='Two Rock'

;;Read in the data file
case ch_sample of
   'Two Rock': file='two_rock.txt'
   'Frei'    : file='frei.txt'
   'Laguna Upper':file='laguna_upper.txt'
endcase 

;;Read the information in the files
readcol,file,date,time,temp,pp,wetness,RH,skipline=1,format='A,A,D,D,D,D'

;;Initiate the date
start_date=date[0]
;;This counts how many different days are measured.
date_check=1 ;;This checks if I am still on the same date
nr_days=0l   
for i=0l,n_elements(date)-1,1 do begin
   if date[i] ne start_date then begin
      nr_days=nr_days+1
      start_date=date[i]
   endif
   
   ;;This account for the last day
   if i eq n_elements(date)-1 then begin
      nr_days++
   endif
endfor

;;Holds the count for the various temperature bins. 
;;index 0 corresponds to the temperatures below <60 and so on 
count_temp=dindgen(9)*0.0 
;;Below are the various indices that hold the other variables. These
;;have to averaged over the daily number of measurements
count_humidity=0.0
count_wetness=0.0
count_precip=0.0
count_measure=0.     ;;This calculates how many measurements there are during each day

;;Master array holding all the various properties
day_properties=dindgen(nr_days,n_elements(count_temp)+3)*0.0

;;For each day loop over the measurements and record the number of 15
;;min intervals that contain tempeatures in the sweet spot.

date_check=1          ;;Check if I am still on the same date
day_nr=0              ;;Running index for the days
start_date=date[0]
;;Loop over all measurements
for i=0l,n_elements(date)-1,1 do begin
   
   ;;Check if it's still the same day
   if date[i] ne start_date then begin
      ;;Populate the array day_properties (in the end this should contain
      ;;the exact number of days that were measured)
      for k=0,n_elements(count_temp)-1,1 do begin
         day_properties[day_nr,k]=count_temp[k]
      endfor
      
      day_properties[day_nr,n_elements(count_temp)]=count_humidity/count_measure
      day_properties[day_nr,n_elements(count_temp)+1]=count_wetness/count_measure
      day_properties[day_nr,n_elements(count_temp)+2]=count_precip/count_measure

      ;;Reset the arrays for the new day and assign the new day for the day checker
      count_temp=count_temp*0.0
      count_humidity=count_humidity*0.0
      count_wetness=count_wetness*0.0
      count_precip=count_precip*0.0
      count_measure=count_measure*0.0
      
      start_date=date[i]
      day_nr++  ;;Increase the day counter
   endif

   ;;Check the temperature and assign the various counters.
   ;;# of 15 min intervals with T>=95
   ;;# of 15 min intervals with 90<=T<95
   ;;# of 15 min intervals with 85<=T<90
   ;;# of 15 min intervals with 80<=T<85
   ;;# of 15 min intervals with 75<=T<80
   ;;# of 15 min intervals with 70<=T<75
   ;;# of 15 min intervals with 65<=T<70
   ;;# of 15 min intervals with 60<=T<65
   ;;# of 15 min intervals with T<60
   case 1 of
      (temp[i] lt 60) : count_temp[0]++
      (temp[i] ge 60) and (temp[i] lt 65) : count_temp[1]++
      (temp[i] ge 65) and (temp[i] lt 70) : count_temp[2]++
      (temp[i] ge 70) and (temp[i] lt 75) : count_temp[3]++
      (temp[i] ge 75) and (temp[i] lt 80) : count_temp[4]++
      (temp[i] ge 80) and (temp[i] lt 85) : count_temp[5]++
      (temp[i] ge 85) and (temp[i] lt 90) : count_temp[6]++
      (temp[i] ge 90) and (temp[i] lt 95) : count_temp[7]++
      (temp[i] ge 95) : count_temp[8]++
   endcase
;readcol,file,date,time,temp,pp,wetness,RH,skipline=1,format='A,A,D,D,D,D'
   count_humidity=count_humidity+RH[i]
   count_wetness=count_wetness+double(wetness[i])
   count_precip=count_precip+pp[i]
   count_measure++
   
   ;;Check if we are on the last day of the series
   if i eq n_elements(date)-1 then begin
      ;;Populate the array day_properties (in the end this should contain
      ;;the exact number of days that were measured)
      for k=0,n_elements(count_temp)-1,1 do begin
         day_properties[day_nr,k]=count_temp[k]
      endfor
      
      day_properties[day_nr,n_elements(count_temp)]=count_humidity/count_measure
      day_properties[day_nr,n_elements(count_temp)+1]=count_wetness/count_measure
      day_properties[day_nr,n_elements(count_temp)+2]=count_precip/count_measure
      
      ;;Reset the arrays for the new day and assign the new day for the day checker
      count_temp=count_temp*0.0
      count_humidity=count_humidity*0.0
      count_wetness=count_wetness*0.0
      count_precip=count_precip*0.0
      count_measure=count_measure*0.0
      
      start_date=date[i]
      day_nr++  ;;Increase the day counter
   endif
   
endfor

;;This section defines the PMI data intervals available from Scott
case ch_sample of
   'Two Rock': begin
      ;;corresponding days: 11 18 26 02 05 09 12 16 22 29 02 09
      sample_intervals=[11,18,26,33,36,40,43,47,51,55,58,65]
      sample_PMI_intervals=[20,0,0,0,40,0,0,0,0,20,20]
      LAMP=[0,0,0,0,0,1,0,0,0,0,0]
      pre_LAMP=[0,0,0,0,0,0,1,0,0,0,0]
      output_file='two_rock_properties.txt'
      treat_mettle= [0,0,0,0,0,0,0,0,0,1,0]
      treat_oil=    [0,0,1,0,0,0,1,0,0,1,0]
      treat_flint=  [0,0,0,0,0,0,0,0,0,0,0]
      treat_kocide= [0,0,1,0,0,0,1,0,0,0,0]
      treat_sulfur= [0,0,0,0,0,0,0,0,0,0,0]
      treat_sylcoat=[0,0,0,0,0,0,0,0,0,0,0]
      treat_switch= [0,0,0,0,0,0,0,0,0,0,0]
      treat_sovran= [0,0,0,0,0,0,0,0,0,0,0]
   end
   'Frei': begin
  ;;corresponding days: 11 18 26 02 05 09 12 16 22 29 02 09 17 
      sample_intervals=[11,18,26,33,36,40,43,47,51,55,58,65,73]
      sample_PMI_intervals=[100,100,80,40,70,70,50,100,70,30,20,20]
      LAMP=[0,0,0,0,0,1,0,0,0,0,0,0]
      pre_LAMP=[0,0,0,0,0,0,1,0,0,0,0,0]
      output_file='frei_properties.txt'
      treat_mettle= [0,0,0,0,0,0,0,1,0,0,0,0]
      treat_oil=    [0,0,0,0,0,0,0,1,0,0,0,1]
      treat_flint=  [0,0,0,0,0,0,0,0,0,0,0,1]
      treat_kocide= [0,0,0,0,0,0,0,0,0,0,0,0]
      treat_sulfur= [0,0,0,0,0,0,0,0,0,0,0,0]
      treat_sylcoat=[0,0,0,0,0,0,0,0,0,0,0,0]
      treat_switch= [0,0,0,0,0,0,0,0,0,0,0,0]
      treat_sovran= [0,0,0,0,0,0,0,0,0,0,0,0]
   end
   'Laguna Upper': begin
  ;;corresponding days: 13 20 27 03 10 17 24 29 06 08 15 22 29 
      sample_intervals=[13,20,27,34,41,48,55,60,67,69,76,83,90]
      sample_PMI_intervals=[0,0,0,20,20,60,60,10,0,70,100,100]
      LAMP=[0,0,0,0,0,0,0,1,1,1,1,1]
      pre_LAMP=[0,0,0,0,0,0,0,0,1,1,1,1]
      output_file='laguna_upper_properties.txt'
      treat_mettle= [0,0,0,0,0,0,0,0,1,0,0,0]
      treat_oil=    [0,0,0,0,0,0,1,0,0,0,0,0]
      treat_flint=  [0,0,0,0,0,0,0,0,0,0,0,0]
      treat_kocide= [0,0,0,0,0,0,1,0,0,0,0,0]
      treat_sulfur= [0,0,0,0,0,0,0,0,1,0,0,0]
      treat_sylcoat=[0,0,0,0,0,0,0,0,1,0,0,0]
      treat_switch= [0,0,0,0,0,0,0,0,0,0,0,0]
      treat_sovran= [0,0,0,0,0,0,0,0,0,0,0,0]
   end
endcase

gd_pmi,PMI_daily,ch_sample
n_int=n_elements(sample_intervals) ;;Variable used to keep the number of intervals provided
;count_treatment=dindgen(n_int-1,8)*0.0
;sample_period=indgen(n_elements(PMI_daily))+1 ;;The days that have data (assumed sequential but may be missing days, need to have that checked)
;sample_PMI=indgen(n_elements(PMI_daily))*0    ;;This is the provided sample PMI
sample_properties=dindgen(n_elements(sample_PMI_intervals),n_elements(count_temp)+6+8)

;;Record the PMI according to the revelant intervals and values
m_val=0
for i=0,n_int-2,1 do begin
   m_val=i
   if i eq n_int-2 then begin
      ;sample_PMI[sample_intervals[i]-1:sample_intervals[i+1]-1]=sample_PMI_intervals[i]
      for k=0,n_elements(count_temp)-1,1 do begin
         sample_properties[m_val,k]=total(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-1,k])
      endfor
      sample_properties[m_val,n_elements(count_temp)]=mean(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-1,n_elements(count_temp)])
      sample_properties[m_val,n_elements(count_temp)+1]=mean(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-1,n_elements(count_temp)+1])
      sample_properties[m_val,n_elements(count_temp)+2]=mean(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-1,n_elements(count_temp)+2])
      sample_properties[m_val,n_elements(count_temp)+3]=mean(PMI_daily[sample_intervals[i]-1:sample_intervals[i+1]-1])
      sample_properties[m_val,n_elements(count_temp)+4]=treat_mettle[m_val]
      sample_properties[m_val,n_elements(count_temp)+5]=treat_oil[m_val]
      sample_properties[m_val,n_elements(count_temp)+6]=treat_flint[m_val]
      sample_properties[m_val,n_elements(count_temp)+7]=treat_kocide[m_val]
      sample_properties[m_val,n_elements(count_temp)+8]=treat_sulfur[m_val]
      sample_properties[m_val,n_elements(count_temp)+9]=treat_sylcoat[m_val]
      sample_properties[m_val,n_elements(count_temp)+10]=treat_switch[m_val]
      sample_properties[m_val,n_elements(count_temp)+11]=treat_sovran[m_val]
      sample_properties[m_val,n_elements(count_temp)+12]=pre_LAMP[m_val]
      sample_properties[m_val,n_elements(count_temp)+13]=LAMP[m_val]
   endif else begin
      ;sample_PMI[sample_intervals[i]-1:sample_intervals[i+1]-2]=sample_PMI_intervals[i]
      for k=0,n_elements(count_temp)-1,1 do begin
         sample_properties[m_val,k]=total(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-2,k])
      endfor
      sample_properties[m_val,n_elements(count_temp)]=mean(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-2,n_elements(count_temp)])
      sample_properties[m_val,n_elements(count_temp)+1]=mean(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-2,n_elements(count_temp)+1])
      sample_properties[m_val,n_elements(count_temp)+2]=mean(day_properties[sample_intervals[i]-1:sample_intervals[i+1]-2,n_elements(count_temp)+2])
      sample_properties[m_val,n_elements(count_temp)+3]=mean(PMI_daily[sample_intervals[i]-1:sample_intervals[i+1]-2])
      sample_properties[m_val,n_elements(count_temp)+4]=treat_mettle[m_val]
      sample_properties[m_val,n_elements(count_temp)+5]=treat_oil[m_val]
      sample_properties[m_val,n_elements(count_temp)+6]=treat_flint[m_val]
      sample_properties[m_val,n_elements(count_temp)+7]=treat_kocide[m_val]
      sample_properties[m_val,n_elements(count_temp)+8]=treat_sulfur[m_val]
      sample_properties[m_val,n_elements(count_temp)+9]=treat_sylcoat[m_val]
      sample_properties[m_val,n_elements(count_temp)+10]=treat_switch[m_val]
      sample_properties[m_val,n_elements(count_temp)+11]=treat_sovran[m_val]
      sample_properties[m_val,n_elements(count_temp)+12]=pre_LAMP[m_val]
      sample_properties[m_val,n_elements(count_temp)+13]=LAMP[m_val]
   endelse
endfor

;;Now output the array sample_properties in the relevant file

si_sample=size(sample_properties)
openw,lun,output_file,/get_lun
;;# of 15 min intervals with T<60
;;# of 15 min intervals with 60<=T<65
;;# of 15 min intervals with 65<=T<70
;;# of 15 min intervals with 70<=T<75
;;# of 15 min intervals with 75<=T<80
;;# of 15 min intervals with 80<=T<85
;;# of 15 min intervals with 85<=T<90
;;# of 15 min intervals with 90<=T<95
;;# of 15 min intervals with T>=95
;;Avg Humidity
;;Av leaf wetness
;;Av precipitation
;;Av_PMI 
;;Mettle   ;;This will record the spraying method used in the previous interval
;;Oil
;;Flint
;;Kocide
;;Sulfur
;;Sylcoat
;;Switch
;;Sovran
;;Previous LAMP
;;LAMP
header='T<60 60<=T<65 65<=T<70 70<=T<75 75<=T<80 80<=T<85 85<=T<90 90<=T<95 T>=95 Av_Hum Av_leaf_wet Av_precip Av_PMI Treat_Mettle Treat_Oil Treat_Flint Treat_Kocide Treat_Sulfur Treat_Sylcoat Treat_Switch Treat_Sovran Prev_LAMP LAMP'
printf,lun,header
for i=0,si_sample[1]-1,1 do begin
   line_string=''
   for j=0,si_sample[2]-1,1 do begin
   line_string=line_string+' '+strtrim(sample_properties[i,j],2)
   endfor
   printf,lun,line_string
endfor
free_lun,lun

;;This plots our calculated PMI 
;tt=plot(sample_period,PMI_daily,sym='*',name='Our PMI')
;;This overplots the provided PMI
;tt1=plot(sample_period[sample_intervals[0]-1:sample_intervals[n_int-1]-1],sample_PMI[sample_intervals[0]-1:sample_intervals[n_int-1]-1],sym='+',color='red',/overplot,/current,name='Provided PMI')
;tt.xtitle='Day record'
;tt.ytitle='PMI'
;tt.title=ch_sample+' location'
;leg = LEGEND(TARGET=[tt,tt1], POSITION=[0.2,0.8],/normal, /AUTO_TEXT_COLOR)


end


function f_t_pmi,pmi

case 1 of 
   (pmi ge 0) and (pmi le 30) : t_pmi=15.0
   (pmi ge 40) and (pmi le 50): t_pmi=10.0
   (pmi ge 60) and (pmi le 100): t_pmi=5.0
endcase

return,t_pmi
end

pro gd_pmi_3,c_measure
;;Generate the number of spores collected by the trap assuming
;;sampling every 2 days. 

;;These are initiating variables
;;This chooses which data table to use.
ch_sample='Laguna Upper'
B=2.0   ;;This is the growth factor for the pathogen. 
measure_freq=2  ;;Measuring frequency (in days) So 2 means every 2 days with no breaks
spray_corr=0.5  ;;This is the efficiency of the spray treatment. Determines by what fraction to reduce the numbers
c_start=1.0     ;;Starting concentration

;;This procedure calculates the PMI from the weather data file chosen
;;based on which sample I want
gd_pmi,PMI_daily,ch_sample

n_days=n_elements(PMI_daily)  ;;Number of days of weather data
c_day_spore=dblarr(n_days+1)  ;;Spore concentrations at the end of each day (plus original concentration)
c_day_int_spore=dblarr(n_days) ;;Integrated spore counts over each day (used to calculate the actual measurements)
c_measure=dblarr(n_days) ;;Concentration measurements. This is the principal output

spray_value=dblarr(n_days) ;;Array indicating if a treatment took place (0-NO,1-YES)
;;This assigns values where the spraying is done. Right now it is
;;inspired by the data sent by scott and the spraying takes place at
;;the end of each of the measuring periods he uses. But it's arbitrary.
case ch_sample of
   'Two Rock': begin
  ;;corresponding days: 11 18 26 02 05 09 12 16 22 29 02 09
      sample_intervals=[11,18,26,33,36,40,43,47,51,55,58,65] ;;This is the locations in the array where the measurements start and end.
      spray_value[25]=1.0
      spray_value[42]=1.0
      spray_value[54]=1.0
   end
   'Frei': begin
  ;;corresponding days: 11 18 26 02 05 09 12 16 22 29 02 09 17 
      sample_intervals=[11,18,26,33,36,40,43,47,51,55,58,65,73]
      spray_value[46]=1.0
      spray_value[64]=1.0
   end
   'Laguna Upper': begin
  ;;corresponding days: 13 20 27 03 10 17 24 29 06 08 15 22 29 
      sample_intervals=[13,20,27,34,41,48,55,60,67,69,76,83,90]
      spray_value[54]=1.0
      spray_value[66]=1.0
      spray_value[89]=1.0
   end
endcase


;;Initiate the c_measure array indicating if a measurement took place. 
;;0 - no measurement 
;;1 - measurement
for i=0,n_days-1,1 do begin
if ((i+1) mod measure_freq eq 0) then c_measure[i]=1.0
endfor

;;Populate the c_day_spore and c_day_int_spore arrays for all the days
c_day_spore[0]=c_start     ;;Original concentration
for i=1,n_days,1 do begin
;;This is the growth time parameter measured daily
   t_pmi=f_t_pmi(PMI_daily[i-1]) 
   
;;This is the spore count at the end of day i. 
;;Also takes into account if there was a spraying done the previous
;;day. If so then the previous day value is corrected by
;;(1-spray_corr) fraction.
   c_day_spore[i]=B^(1.0/t_pmi)*c_day_spore[i-1]*(1.0-spray_corr*spray_value[i-1])  
   
;;This is the integrated spore count at the end of day i.
   c_day_int_spore[i-1]=t_pmi/alog(B)*(c_day_spore[i]-c_day_spore[i-1]*(1.0-spray_corr*spray_value[i-1]))  
   
endfor 

;;Calculate the concentration measured at the end of each measuring
;;period by adding up the integrated spore counts over the days the
;;trap is outside
c_val=0.0 ;;Dummy variable to keep track of concentration
for i=0,n_days-1,1 do begin
   c_val=c_val+c_day_int_spore[i]

   ;;Check if this is the day that a measurement is taken (assume the
   ;;measurement is taken at end of day)
   if c_measure[i] gt 1.e-5 then begin 
      c_measure[i]=c_val        ;;Record the measurement
      c_val=0.0                 ;;Reset the dummy variable
   endif
endfor

end
