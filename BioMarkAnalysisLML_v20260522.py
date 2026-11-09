
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, linregress
import os
import seaborn as sns

class BioMarkAnalysisLML:
    
    def __init__(self, raw_fluorescence, plate_layout_file, chip_layout_file, experiment_name):
        """Initialize with raw fluorescence and plate layout inputs.
        
        Args:
           raw_fluorescence: raw fluorescence data in txt file
           plate_layout: plate layout in 96 well format in txt file 
            experiment_name: name to use for saving files 
            
            
        """
        self.raw = raw_fluorescence #list of dataframes 
        self.raw_fam = raw_fluorescence['probe_raw']
        self.raw_rox = raw_fluorescence['ref_raw']
        self.plate_layout_file = plate_layout_file
        self.experiment_name = experiment_name 
        self.output_path = f"{self.experiment_name}_{date.today()}_analysis"
        self.figure_output_path = f"{self.experiment_name}_{date.today()}_analysis/figures/"
        self.chip_layout_file = chip_layout_file
        self.chamber_ids = self.raw_fam.index
        
    def create_output_directory(self):
        try:
            # Create target directory & all intermediate directories if they don't exist
            os.makedirs(self.output_path, exist_ok=True)
            print(f"Output directory '{self.output_path}' created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")
            
        try:
            # Create target directory & all intermediate directories if they don't exist
            os.makedirs(self.figure_output_path, exist_ok=True)
            print(f"Output directory '{self.figure_output_path}' created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")
            
            
        return self.output_path
            
    def plate_layout_to_chip_layout(self):
        # --- Inner well layout ---
        inner_left = pd.read_excel(self.plate_layout_file, sheet_name="inner_left", header=None)
        inner_right = pd.read_excel(self.plate_layout_file, sheet_name="inner_right", header=None)
    
        if inner_left.shape != (8, 12) or inner_right.shape != (8, 12):
            print("Error! Inner plate layout has incorrect shape")
            print(f"inner_left shape: {inner_left.shape}, inner_right shape: {inner_right.shape}")
        else:
            # IFC is 16 rows x 12 cols
            # Left plate (inner_left) fills IFC cols 0-5, right plate (inner_right) fills IFC cols 6-11
            # 
            # Plate col loading order into IFC columns:
            #   Plate cols 1-6 (0-indexed: 0-5) fill IFC odd rows (0,2,4,6,8,10,12,14)
            #   Plate cols 7-12 (0-indexed: 6-11) fill IFC even rows (1,3,5,7,9,11,13,15)
            #   IFC column order for both: 1,3,5,2,4,6 (0-indexed: 0,2,4,1,3,5)
            
            ifc_col_order = [0, 2, 4, 1, 3, 5]  # plate cols 1-6 and 7-12 map to IFC cols in this order
            
            def plate_to_ifc_half(plate):
                # Build a 16 x 6 array
                ifc_half = np.empty((16, 6), dtype=object)
                for plate_col_idx in range(6):
                    ifc_col = ifc_col_order[plate_col_idx]
                    for plate_row in range(8):
                        # Plate cols 0-5 go into odd IFC rows (0,2,4,6,8,10,12,14)
                        ifc_row = plate_row * 2
                        ifc_half[ifc_row, ifc_col] = plate.iloc[plate_row, plate_col_idx]
                for plate_col_idx in range(6):
                    ifc_col = ifc_col_order[plate_col_idx]
                    for plate_row in range(8):
                        # Plate cols 6-11 go into even IFC rows (1,3,5,7,9,11,13,15)
                        ifc_row = plate_row * 2 + 1
                        ifc_half[ifc_row, ifc_col] = plate.iloc[plate_row, plate_col_idx + 6]
                return ifc_half
            
            left_half = plate_to_ifc_half(inner_left)
            right_half = plate_to_ifc_half(inner_right)
            
            # Combine: left half is IFC cols 0-5, right half is IFC cols 6-11
            self.inner_layout = pd.DataFrame(np.hstack([left_half, right_half]))
            print("Inner layout generated")
    
        # --- Outer well layout ---
        outer_layout = pd.read_excel(self.plate_layout_file, sheet_name="outer", header=None)
    
        if outer_layout.shape != (8, 3):
            print("Error! Outer layout has incorrect shape")
            print(f"outer shape: {outer_layout.shape}")
        else:
            self.outer_layout = outer_layout
            print("Outer layout generated")
        
    def layout_to_annotations(self): 
        inner_chip = pd.read_excel(self.chip_layout_file, sheet_name = 'inner', header = None)
        outer_chip = pd.read_excel(self.chip_layout_file, sheet_name = 'outer', header = None)
        
        self.S_dict = dict(zip(inner_chip.values.flatten(), self.inner_layout.values.flatten()))
        self.A_dict = dict(zip(outer_chip.values.flatten(), self.outer_layout.values.flatten()))
        
    
        self.annotation_dict = {}


        for chamber in self.chamber_ids :

            annots = {} #dict for each chamber

            annots['S_well'] = chamber.split("-")[0]
            annots['A_well'] = chamber.split("-")[1]

            annots['inner'] = self.A_dict[annots['A_well']]
            annots['outer'] = self.S_dict[annots['S_well']]

            #standard annotations
            annots['master_mix'] = self.S_dict[annots['S_well']].split('_')[0]
            annots['primer_set'] = self.S_dict[annots['S_well']].split('_')[1]

            annots['target_seq'] = self.A_dict[annots['A_well']].split('_')[0]
            annots['target_conc'] = self.A_dict[annots['A_well']].split('_')[1]
            annots['target_conc_float'] = float(self.A_dict[annots['A_well']].split('_')[1])
            annots['target_type'] = self.A_dict[annots['A_well']].split('_')[2]

            self.annotation_dict[chamber] = annots

        print(f'Annotations created for {len(self.annotation_dict)} chambers')
        
        for val_type in ['master_mix','primer_set','target_seq']:
            print(f'Annotated {val_type}: {set([value.get(val_type) for key, value in self.annotation_dict.items()])}')
            
         #make a mastermix color dict 
        MMs = list(set(value.get('master_mix') for key, value in self.annotation_dict.items()))
        self.MM_color_dict = {MM:color for MM, color in zip(MMs, sns.color_palette('Set2',len(MMs)))}
        
        return self.annotation_dict
    
            
    def find_replicate_chambers(self) :
        replicate_dict = {}
        for key, value in self.annotation_dict.items() :
            combo = (value.get('master_mix'), value.get('primer_set'), value.get('target_seq'))
            if combo in replicate_dict : 
                replicate_dict[combo].append(key)
            else : 
                replicate_dict[combo] = [key]
        
        pruned_dict = {key : value for key, value in replicate_dict.items() if len(value) > 1}
        
        print(f"Found {len(pruned_dict)} replicate groups with > 1 replicate")
        
        return pruned_dict
        
    def plot_ROX(self, plot = False):
        data = self.raw_rox
        data["chamber_id"] = data.index
        data[['S_well','A_well']] = data["chamber_id"].str.split("-", expand = True)

        # Pivot the DataFrame
        reshaped_df = data.pivot_table(index='S_well', columns='A_well', values='t1')

        if plot == True :
            #Plot it
            plt.subplots(1,1, figsize = (20,18))
            sns.heatmap(reshaped_df.transpose(),square = True, cmap = "gist_yarg", cbar_kws = {'location':'bottom'}, 
                        vmin = np.min(reshaped_df), vmax = np.max(reshaped_df))
            plt.title(f"Loading Check: Raw ROX at t1\n{self.experiment_name}")

        #Make a list of S_wells and A_wells that failed loading
        self.loading_failed_samples = [] #will use these lists to filter our fam dataframe later
        self.loading_failed_assays = []

        #set threshold as mean - 3*std
        threshold = reshaped_df.values.flatten().mean() - 3 * (reshaped_df.values.flatten().std())

        for samp in reshaped_df.index :
            count = (reshaped_df.loc[samp] < threshold).sum()
            if count > 8 :
                self.loading_failed_samples.append(samp)
                print(f"Sample well {samp} failed loading")

        for assay in reshaped_df.columns :
            count = (reshaped_df.loc[:, assay] < threshold).sum()
            if count > 64 :
                self.loading_failed_assays.append(assay)
                print(f"Assay well {assay} failed loading")
                
    def format_data(self, min_max_norm = True, monotonic_smooth = False, baseline_norm = False, ROXnorm = False, interval = 1, n_baseline_cols = 3) :
        #drop chambers that failed loading 
        chambers_to_drop = []
    
        for samp in self.loading_failed_samples:
            samp_chambers = [samp + "-" + x for x in self.A_dict.keys()]
            chambers_to_drop += samp_chambers
    
        #do the opposite for assays
        for assay in self.loading_failed_assays:
            assay_chambers = [x + "-" + assay for x in self.S_dict.keys()]
            chambers_to_drop += assay_chambers
    
        print(f"{len(chambers_to_drop)} chambers were not included in analysis")
    
        print(f"The following target seqs were dropped: {[str(x) + ': ' + str(self.A_dict[x]) for x in self.loading_failed_assays]}")
        print(f"The following primer sets were dropped: {[str(x) + ': ' + str(self.S_dict[x]) for x in self.loading_failed_samples]}")
        
        #get the correct data time 
        #fill in the interval for your time points in minutes (ie 30s = 0.5)
        interval = interval
        self.time_points = self.raw_fam.columns
        self.time_points_int = [int(tpoint.strip('t'))*interval for tpoint in self.time_points]
    
        chambers_to_include = [chamber for chamber in self.chamber_ids if chamber not in chambers_to_drop]
        
        #first normalize to rox
        if ROXnorm == True : 
            data = self.raw_fam[self.time_points].div(self.raw_rox[self.time_points]) 
        else :
            data = self.raw_fam.copy()
    
        #monotonic smoothing: enforce non-decreasing signal from left to right
        if monotonic_smooth == True :
            def remove_noise(row):
                vs = row.values[::-1]
                newvs = [vs[0]]
                lmin = vs[0]
                for i in range(1, len(vs)):
                    if vs[i] < lmin:
                        lmin = vs[i]
                    newvs.append(lmin)
                return pd.Series(newvs[::-1], index=row.index)
            
            data = data.apply(remove_noise, axis=1)
    
        #baseline normalization: subtract baseline and clip negatives to zero
        if baseline_norm == True :
            baseline = data.iloc[:, 0:n_baseline_cols].mean(axis=1)
            data = data.subtract(baseline, axis=0).clip(lower=0)
    
        #min max normalization: baseline subtract and scale to max per master mix
        if min_max_norm == True :
            #if baseline_norm already ran, skip the baseline subtraction
            if baseline_norm == False :
                data = data.subtract(data.iloc[:,0:n_baseline_cols].mean(axis = 1), axis = 0)
                
            #must calculate the maximum for each master mix, then merge together 
            MM_data= []
            for MM in self.MM_color_dict.keys() : 
                MM_keys = [key for key, value in self.annotation_dict.items() if 
                    value.get('master_mix') == MM]
                MM_data.append(data.loc[MM_keys].div(np.max(data.loc[MM_keys])) )
          
            data = pd.concat(MM_data)
    
        self.experimental_data = data
    
        print(f"experimental_data dataset contains {len(self.experimental_data.index)} samples and\
         {len(self.experimental_data.columns)} timepoints")
    
        
         #save so that they will be retained in the pickle object 
        self.ROXnorm = ROXnorm
        self.monotonic_smooth = monotonic_smooth
        self.baseline_norm = baseline_norm
        self.min_max_norm = min_max_norm
        self.n_baseline_cols = n_baseline_cols
        
        print(f"ROX normalization: {self.ROXnorm}")
        print(f"Monotonic smoothing: {self.monotonic_smooth}")
        print(f"Baseline normalization: {self.baseline_norm}")
        print(f"Min max normalization: {self.min_max_norm}")
        
       
        
        return self.experimental_data
        
    def plot_background(self):
        #check if background is rising over time
        fig, (ax1,ax2) = plt.subplots(1,2, figsize = (8,3))
        mean_fam_bkgd = [np.mean(self.raw['probe_bkgd'].loc[:,t]) for t in self.time_points]
        std_fam_bkgd = [np.std(self.raw['probe_bkgd'].loc[:,t]) for t in self.time_points]

        mean_rox_bkgd = [np.mean(self.raw['ref_bkgd'].loc[:,t]) for t in self.time_points]
        std_rox_bkgd = [np.std(self.raw['ref_bkgd'].loc[:,t]) for t in self.time_points]

        ax1.errorbar(x = self.time_points, y = mean_fam_bkgd, yerr = std_fam_bkgd, fmt = 'o', color = "red")
        ax1.set_title(f"FAM background over run\n{self.experiment_name}")

        ax2.errorbar(x = self.time_points, y = mean_rox_bkgd, yerr = std_rox_bkgd, fmt = 'o')
        ax2.set_title(f"ROX background over run\n{self.experiment_name}")
        
    def plot_fold_change_raw_fam(self, plottype = 'hist'):

        master_mixes = sorted(list({value.get('master_mix') for key, value in self.annotation_dict.items()}))
        
        data_list = []
        
        for MM in master_mixes : 
            keys = [key for key, value in self.annotation_dict.items()
                   if value.get('master_mix') == MM]# and value.get('primer_set') == "m13"]
            
            data_t1 = dataframes['probe_raw'].loc[keys, "t1"]
            data_t60 = dataframes['probe_raw'].loc[keys, "t60"]
            data_list.append( data_t60 / data_t1)

        # List of datasets for stacked hist
    
        colors = [self.MM_color_dict[MM] for MM in master_mixes] 
        labels = master_mixes

        

        if plottype == "hist" :
            plt.figure(figsize=(5,4))
            #Stacked histogram
            plt.hist(data_list, bins=30, stacked=True, color=colors, label=labels, alpha=0.7, edgecolor='black')
            
            plt.xlabel('Raw FAM t60 / t1')
            plt.ylabel('Number of chambers')
            plt.legend()
            plt.tight_layout()
            plt.title("Distribution of FAM fold change for all reactions")
                      
        elif plottype == "kde" :
            fig, ax = plt.subplots(1,1, figsize=(5,4))
            
            #KDE curves for each
            for MM, data, color in zip(master_mixes, data_list, colors) : 
                sns.kdeplot(data, color=color, label=MM, ax= ax)
            
            plt.xlabel('Raw FAM t60 / t1')
            plt.ylabel('density')
            plt.legend()
            plt.tight_layout()
            plt.title("Distribution of FAM fold change for all reactions")
        
    
    def find_and_plot_NTCs(self, plot = True, savefig = False) : 
        #get tupple of all MM/PS combos 
        self.NTC_keys = {} 
        MM_PS = {(value.get('master_mix'),value.get('primer_set')) for key, value in self.annotation_dict.items()}
        
        for combo in MM_PS: 
            combo_keys = [key for key, value in self.annotation_dict.items() if
            "NTC" in value.get('target_seq') and 
            value.get('master_mix') == combo[0] and
            value.get('primer_set') == combo[1]]

            self.NTC_keys[combo] = combo_keys # save the NTC keys  
        
        return self.NTC_keys
    
        if plot == True :
            
            primer_sets = sorted(list({value.get('primer_set') for key, value in self.annotation_dict.items()}))
            master_mixes = sorted(list({value.get('master_mix') for key, value in self.annotation_dict.items()}))
            
            n_rows = int(np.ceil(len(primer_sets) /6))
            fig, axes = plt.subplots(n_rows,6 , figsize=(12, 2*n_rows), sharey = True, sharex = True)
            fig.tight_layout(h_pad = 3, w_pad = 1 )

            for pset,ax in zip(primer_sets,axes.ravel()) : 
                for MM, color in self.MM_color_dict.items() : #zip(master_mixes,['lightseagreen','thistle']): 
                    if (MM,pset) in self.NTC_keys :
                        keys = self.NTC_keys[(MM,pset)]
                        for key in keys : 
                            ax.plot(self.time_points_int, self.experimental_data.loc[key,:], color = color)
                
                ax.set_ylim(np.min(self.experimental_data),np.max(self.experimental_data))
                ax.set_title(f"NTC amplification:\n{pset}")
                
                ax.legend(handles = [mpatches.Patch(label = l, color = c) for l,c in  self.MM_color_dict.items()])#zip(master_mixes,['lightseagreen','thistle'])])
                
            fig.suptitle(f"Non-template control amplification for all primer sets and master mixes\n{self.experiment_name}", y = 1.1) 
        
        if savefig == True : 
            
            fig.savefig(f"{self.figure_output_path}/NTC_amp_curves.pdf", format = "pdf", bbox_inches = 'tight')
            fig.savefig(f"{self.figure_output_path}/NTC_amp_curves.png", format = "png", bbox_inches = 'tight')
            
            
    def plot_internal_control(self, pos_cntrl_pset = "m13", plot = True, savefig = False) : 
        target_seqs = sorted(list({value.get('target_seq') for key, value in self.annotation_dict.items()}))
        master_mixes = sorted(list({value.get('master_mix') for key, value in self.annotation_dict.items()}))
            
        n_rows = int(np.ceil(len(target_seqs)/6))
        fig, axes = plt.subplots(n_rows,6 , figsize=(12, 2*n_rows), sharey = True, sharex = True)
        fig.tight_layout(h_pad = 3, w_pad = 1 )

        for targ,ax in zip(target_seqs,axes.ravel()) : 
            for MM, color in  self.MM_color_dict.items(): #zip(master_mixes,['lightseagreen','thistle']): 
                keys = [key for key, value in self.annotation_dict.items() if
                    value.get('target_seq') == targ and 
                    value.get('master_mix') == MM and
                    value.get('primer_set') == pos_cntrl_pset]
                
                for key in keys : 
                    ax.plot(self.time_points_int, self.experimental_data.loc[key,:], color = color)

            ax.set_ylim(np.min(self.experimental_data),np.max(self.experimental_data))
            ax.set_title(f"Int. control amp:\n{targ}")

            ax.legend(handles = [mpatches.Patch(label = l, color = c) for l,c in self.MM_color_dict.items()])
        fig.suptitle(f"Internal positive control amplification for all primer sets and master mixes\n{self.experiment_name}", y = 1.1) 
            
        if savefig == True : 
            
            fig.savefig(f"{self.figure_output_path}/m13mp18_amp_curves.pdf", format = "pdf", bbox_inches = 'tight')
            fig.savefig(f"{self.figure_output_path}/m13mp18_amp_curves.png", format = "png", bbox_inches = 'tight')
            
    def plot_all_targets(self) : 
        #get tupple of all MM/PS combos 
        MM_PS_T = {(value.get('master_mix'),value.get('primer_set')) for key, value in self.annotation_dict.items()}
        
        n_rows = int(np.ceil(len(MM_PS)/4))
        fig, axes = plt.subplots(n_rows,4 , figsize=(16, 4*n_rows), sharey = True, sharex = True)
        fig.tight_layout(h_pad = 3, w_pad = 1 )
        
        targets = sorted({value.get('target_seq') for key, value in self.annotation_dict.items()})
        
        
                
        for combo, ax in zip(MM_PS, axes.ravel()): 
            handles = []
            
            for target, color in zip(targets,sns.color_palette('Spectral', len(targets))) :

                if target == "NTC" :
                    color = "lightgrey"
                    
                handles.append(mpatches.Patch(color=color, label=target))
                    
                keys = [key for key, value in self.annotation_dict.items() if
                    value.get('target_seq') == target and 
                    value.get('master_mix') == combo[0] and
                    value.get('primer_set') == combo[1]]


                ax.errorbar(x = self.time_points_int, y = np.mean(self.experimental_data.loc[keys,:], axis = 0), 
                            yerr = np.std(self.experimental_data.loc[keys,:], axis = 0), color = color,
                           label = target)
            
            ax.set_ylim(np.min(self.experimental_data),np.max(self.experimental_data))

            ax.set_title(f"Target amplification:\n{combo}")
            
        fig.legend(handles = handles)
    
    
    def binary_amplification(self, verbose = True, threshold = 0.2) :
        
        self.amp_metric = "fluor_thresh" #save the amp metric that you used 
        self.amp_threshold = threshold
        
        dict_made = False #set to false to start 
        
       
    
        self.amp_dict = {} 

        for item in self.chamber_ids:
            y_data = self.experimental_data.loc[item,:] #get data for all tiem points


            if np.max(y_data) >= threshold:

                self.amp_dict[item] = 1
            else:
                self.amp_dict[item] = 0
        dict_made = True



        if dict_made == True :

            print(f"{sum(1 for key, value in self.amp_dict.items() if value == 1)} samples amplified")
            print(f"{sum(1 for key, value in self.amp_dict.items() if value == 0)} samples did not amplified")


            if verbose == True : 
                for combo, keys in self.NTC_keys.items() : 
                    ampd = sum(self.amp_dict[key] for key in keys)
                    if ampd == 0 :
                        print(f"No NTC amplification detected for {combo}")
                    else :
                        print(f"{ampd} / {len(keys)} NTCs amplified for {combo}")

            return self.amp_dict

    def fit_generalized_logistic(self, initial_guess = [1, 0, 0.5, 30], bounds = (
                        [0.2, 0, 0.01, 0],  # Lower bounds
                        [1, 0.1, 5, 60]      # Upper bounds - keep 'a' between 0.8-1.2
                    )):
        """function to fit fluorescence amplification data to a generalized logistic curve 

        requires 2 inputs: 
        (1) experimental data containing a column for each time point and a row for each well 
        (2) length of measurement interval in minutes (ie 30s = 0.5)

        this function produces 3 outputs: 
         (1) a dictionary with the fit parameters for each well
         (2) Tp for each well 
         (3) a dictionary with the R-squared value for the fit to the data and 
         if there is not sufficient evidence of positive amplification, defined as 
         t60 fluorescence > threshold
         the threshold is defined as 3*stddev(NTC) + mean(NTC) but this must be calculated separately 
         then the data will not be fit and the well will not be included in the dicts
         if the logistic fit times out, the well will not be included in the dicts
         if the logistic fit is not good (R2 < 0.9), the well will not be included in the dicts"""

        
        
        # Define the generalized logistic function to not allow negative a we will use to fit
        def generalized_logistic(t, k, a, b, m):
            return max(0, a) + (k - max(0, a)) / (1 + np.exp(-b * (t - m)))

        # Function to find R-squared value for goodness of fit
        def r_squared(observed, predicted):
            """Calculate the coefficient of determination (R-squared value)."""
            residuals = observed - predicted
            mean_observed = np.mean(observed)
            tss = np.sum((observed - mean_observed) ** 2)
            rss = np.sum((observed - predicted) ** 2)
            r_squared = 1 - (rss / tss)
            return r_squared

        self.fit_params_dict = {}  # Create a dict to store the fit params for each well
        self.Tp_dict = {}  # Create a dict to store the Tp results for each sample
        self.rsquared_dict = {}  # Create a dict to store R-squared values
        self.fit_category = {}
        
        failed_amp_count = 0 #count number that fail for amp
        failed_R_count = 0 #count number that fail for R
        failed_fit_count = 0

        self.guess = initial_guess 
        
        self.bounds = bounds
            
            
        # Transform your time points into integers for the math, adjust for interval in minutes
        x_data = self.time_points_int
        for item in self.chamber_ids:  # For each index (ie each chamber) calculate fit
            if self.amp_dict[item] == True :
                y_data = self.experimental_data.loc[item,:]

                try:
                     # Fit your data to the model

                    params, covariance = curve_fit(
                        generalized_logistic, x_data, y_data, 
                        p0=self.guess, bounds=self.bounds, maxfev=1000
                    )
                    # Fit your data to the model
                    
                    rsquared = r_squared(y_data, generalized_logistic(x_data, *params))
                    self.rsquared_dict[item] = rsquared

                    # Check if R-squared is ≥ 0.9
                    if rsquared < 0.9:
                        self.fit_category[item] = "low_r_squared_fit"
                        failed_R_count += 1 
                        pass
                    else:
                        self.fit_params_dict[item] = dict( zip(["k","a","b","m"],params))
                        self.Tp_dict[item] = params[3] - (2 / params[2])
                        self.fit_category[item] = "good_fit"

                except RuntimeError:
                    self.fit_category[item] = "unable_to_fit"
                    failed_fit_count +=1 
                    pass  #Exit the loop if there's no good fit for the sample
            else :
                self.fit_category[item] = "no_amp"
                pass 

        print(f"{sum(1 for value in self.Tp_dict.values())} / {len(self.chamber_ids)} samples had "
              f" a logistic fit with R2 >= 0.9. Tp reported for these samples")

        print(f"Reasons for no tp: {failed_amp_count} sample had no amplification \n {failed_R_count}: samples had Rsquared < 0.9 \n {failed_fit_count}: samples had no logistic fit")
        
        print(f"Bounds: {self.bounds}")
        
        return self.Tp_dict , self.fit_params_dict, self.rsquared_dict
    
    def threshold_tp_by_NTC(self) : 
        #Define the Tp threshold for each assay 
        self.tp_max = {} 
        
        MM_PS = {(value.get('master_mix'),value.get('primer_set')) for key, value in self.annotation_dict.items()}
        
        for combo in MM_PS : 
            
            
            #check if any NTCs amplified
            if sum(1 for key in self.NTC_keys[combo] if self.amp_dict[key] == True) > 0 : 

                NTC_tps = [self.Tp_dict[key] for key in self.NTC_keys[combo] if key in self.Tp_dict]

                self.tp_max[combo] = np.mean(NTC_tps)- 3 * np.std(NTC_tps)

            else : #if nothing amplified, max is last time point 
                self.tp_max[combo] = self.time_points_int[-1]

       #  # now check if each sample meets this threshold 
#         keys = self.experimental_data.index
#         
#         self.amp_thresh_tp_dict = {
#                 key: True if (self.amp_dict[key] is True and self.Tp_dict.get(key, float('inf')) < self.tp_max[(self.annotation_dict[key].get('master_mix'),
#                                                                                                                self.annotation_dict[key].get('primer_set'))]) else False
#                 for key in keys}
    
    def plot_all_targets_tp(self) : 
        #get tupple of all MM/PS combos 
        MM_PS = {(value.get('master_mix'),value.get('primer_set')) for key, value in self.annotation_dict.items()}
        
        n_rows = int(np.ceil(len(MM_PS)/8))
        fig, axes = plt.subplots(n_rows,8 , figsize=(16, 2*n_rows), sharey = True, sharex = True)
        fig.tight_layout(h_pad = 3, w_pad = 1 )
        
        targets = [t for t in sorted({value.get('target_seq') for key, value in self.annotation_dict.items()}) if t != "NTC"]
        
        
                
        for combo, ax in zip(MM_PS, axes.ravel()): 
            handles = []
            xticks = [] 
            for i, (target, color) in enumerate(zip(targets,sns.color_palette('Spectral', len(targets)))) :
                    
                handles.append(mpatches.Patch(color=color, label=target))
                    
                keys = [key for key, value in self.annotation_dict.items() if
                    value.get('target_seq') == target and 
                    value.get('master_mix') == combo[0] and
                    value.get('primer_set') == combo[1] and 
                        key in self.Tp_dict]
        
                
                xticks.append(f"{target}\nn={len(keys)}")


                if len(keys) == 0 : 
                    ax.bar(x = i, height = 0, color = color)
                else : 
                    ax.bar(x = i, height = np.mean([self.Tp_dict[key] for key in keys]),
                       yerr = np.std([self.Tp_dict[key] for key in keys]), color = color)
            
            ax.set_xticks(ticks = range(len(targets)), labels = xticks);
            ax.set_title(f"Mean Tp:\n{combo}")
            
        fig.legend(title = "Target name", handles = handles, bbox_to_anchor = (1.3,1))
        
    def create_summary_tables(self, save_outputs = False,pos_cntrl_pset = "m13-Pexp1") :
        #create chamber df which contains results for each individual biomark chamber 
        
        # chamber_df = pd.DataFrame(data = {'chamber_id':self.chamber_ids,
        #         f't{self.time_points[-1]}':[self.experimental_data.loc[chamber][-1] for chamber in self.chamber_ids],
        #         'target_seq':[self.annotation_dict[chamber].get('target_seq') for chamber in self.chamber_ids],
        #         'target_conc':[self.annotation_dict[chamber].get('target_conc') for chamber in self.chamber_ids],
        #             'primer_set':[self.annotation_dict[chamber].get('primer_set') for chamber in self.chamber_ids],
        #         'master_mix':[self.annotation_dict[chamber].get('master_mix') for chamber in self.chamber_ids],
        #         'tp':[self.Tp_dict[chamber] if chamber in self.Tp_dict else np.nan for chamber in self.chamber_ids],
        #         f'amp_metric_{self.amp_metric}':[self.amp_dict[chamber] for chamber in self.chamber_ids]})

        
        keys =self.chamber_ids
        
        chamber_data_df = pd.DataFrame(data = {'key':keys,
                f'amp_metric_{self.amp_metric}':[self.amp_dict[key] for key in keys],
                'fit_category':[self.fit_category[key] if key in self.fit_category else np.nan for key in keys],
                'r_squared':[self.rsquared_dict[key] if key in self.rsquared_dict else np.nan for key in keys ],
                'tp':[self.Tp_dict[key] if key in self.Tp_dict else np.nan for key in keys ],
                'k':[self.fit_params_dict[key].get('k') if key in self.fit_params_dict else np.nan for key in keys],
                'a':[self.fit_params_dict[key].get('a') if key in self.fit_params_dict else np.nan for key in keys],
                'm':[self.fit_params_dict[key].get('b') if key in self.fit_params_dict else np.nan for key in keys],
                'b':[self.fit_params_dict[key].get('m') if key in self.fit_params_dict else np.nan for key in keys],
                'f':[self.fit_params_dict[key].get('f') if key in self.fit_params_dict else np.nan for key in keys]})
        
        
        chamber_df = pd.DataFrame.from_dict(self.annotation_dict, orient = 'index').merge(chamber_data_df.set_index('key'),
            left_index = True, right_index = True, how = 'left') 
        
        chamber_df['tp_threshold'] = [self.tp_max[combo] for combo in [(MM,PS) for MM, PS in zip(chamber_df.master_mix, chamber_df.primer_set)]]
        chamber_df['amp_thresh_tp'] = chamber_df['tp'] < chamber_df['tp_threshold']
        
        # Group the DataFrame by target seq, primer set, master mix and calculate summary statistics
        grouped = chamber_df.groupby(['target_seq','target_conc','primer_set','master_mix'])
    
        amp_metric_column = f'amp_metric_{self.amp_metric}'
        # Create the DataFrame with aggregated results
        result = grouped.agg(          # Number of rows in each group
            nreps = ('target_seq','size'),
            nreps_tp=('tp', 'count'),           # Number of non-null `tp` values
            tp_mean=('tp', 'mean'),             # Average of `tp`
            tp_std=('tp', 'std'),               # Standard deviation of `tp`
            tp_min=('tp', 'min'),   
            tp_max=('tp', 'max'), 
            m_mean=('m', 'mean'),             # Average of `tp`
            m_std=('m', 'std'),               # Standard deviation of `tp`
            m_min=('m', 'min'),   
            m_max=('m', 'max'), # range of tp between replicates 
            reps_amped=(amp_metric_column, 'sum') # Sum of the amplification metric
        ).reset_index()
  

        # add a colum with the amplification metric used 
        result['amplification_metric'] = self.amp_metric
        
        # calculate proportions of replicates 
        result["proportion_reps_tp"] = result.nreps_tp / result.nreps
        result["tp_range"] = result.tp_max - result.tp_min
        result["proportion_reps_amped"] = result.reps_amped / result.nreps
        
        #calculate the delta tp
        pos_cntrl = result.loc[result.primer_set == pos_cntrl_pset][['target_seq', 'master_mix', 'tp_mean']].rename(columns={'tp_mean': 'tp_internal_control'})
        result = result.merge(pos_cntrl, left_on = ['target_seq','master_mix'], right_on = ['target_seq','master_mix'], how = 'left').copy()
        result['delta_tp_mean'] = result['tp_mean'] - result['tp_internal_control']
        
        chamber_df = chamber_df.reset_index(names = 'chamber_id').merge(pos_cntrl, left_on = ['target_seq','master_mix'], right_on = ['target_seq','master_mix'], how = 'left').copy()
        chamber_df = chamber_df.set_index('chamber_id')
        chamber_df['delta_tp'] = chamber_df['tp'] - chamber_df['tp_internal_control']
       
        chamber_df['barcode'] = self.experiment_name
        result['barcode'] = self.experiment_name
            
        if save_outputs == True : 
            chamber_df.to_csv(f"{self.output_path}/{self.experiment_name}_chamber_statistics.csv")
            print(f"Chamber data saved to {self.output_path}/{self.experiment_name}_chamber_statistics.csv")
            
            result.to_csv(f"{self.output_path}/{self.experiment_name}_summary_statistics.csv")
            print(f"Summary data saved to {self.output_path}/{self.experiment_name}_summary_statistics.csv")
        
        self.result = result
        self.chamber_df = chamber_df 
        
        return chamber_df, result
   
                
            
    def heatmap_tp(self, savefig = False) :
        master_mixes = sorted(list({value.get('master_mix') for key, value in self.annotation_dict.items()}))
        
        if len(master_mixes) > 1 : 
            fig, axes = plt.subplots(1,len(master_mixes), figsize = (18,6*len(master_mixes)))
            fig.tight_layout(h_pad = 4, w_pad = 8)

            for MM, ax in zip(master_mixes,axes.ravel()) : 
                data = self.result.query('master_mix == @MM & proportion_reps_amped == 1').pivot(columns = "primer_set", index = ["target_seq","target_conc"], values = 'tp_mean')

                sns.heatmap(data =data, square = True, cmap = "plasma",
                            linewidth = 0.1, linecolor = 'darkgrey',
                           cbar_kws = {'label':f'Mean time to positive',
                                       'location' : 'bottom', 'pad':0.25,'shrink': 0.6},
                           ax = ax)

                ax.set_title(MM)
                ax.set_ylabel("Target sequence")

            fig.suptitle(f"IFC barcode: {self.experiment_name}") 
        
        else :

            fig, ax = plt.subplots(1,1, figsize = (18,6))
            
            MM = master_mixes[0] 
                
            data = self.result.query('master_mix == @MM & proportion_reps_amped == 1').pivot(columns = "primer_set", index = ["target_seq","target_conc"], values = 'tp_mean')

            sns.heatmap(data =data, square = True, cmap = "plasma",
                        linewidth = 0.1, linecolor = 'darkgrey',
                       cbar_kws = {'label':f'Mean time to positive',
                                   'location' : 'bottom', 'pad':0.25,'shrink': 0.6},
                       ax = ax)

            ax.set_title(MM)
            ax.set_ylabel("Target sequence")

            fig.suptitle(f"IFC barcode: {self.experiment_name}") 
        
        if savefig == True : 
            
            fig.savefig(f"{self.figure_output_path}/heatmap_tp.pdf", format = "pdf", bbox_inches = 'tight')
            fig.savefig(f"{self.figure_output_path}/heatmap_tp.png", format = "png", bbox_inches = 'tight')
            
            
            
    def heatmap_reps_amped(self, savefig = False) :
        master_mixes = sorted(list({value.get('master_mix') for key, value in self.annotation_dict.items()}))
        
        if len(master_mixes) > 1 : 
            fig, axes = plt.subplots(1,len(master_mixes), figsize = (18,8*len(master_mixes)))
            fig.tight_layout(h_pad = 4, w_pad = 8)

            for MM, ax in zip(master_mixes,axes.ravel()) : 
                data = self.result.query('master_mix == @MM').pivot(columns = "primer_set", index = ["target_seq","target_conc"], values = 'proportion_reps_amped')

                sns.heatmap(data =data, square = True, cmap = "binary",
                            linewidth = 0.1, linecolor = 'darkgrey',vmin =0, vmax = 1,
                           cbar_kws = {'label':f'Proportion reps amplified',
                                       'location' : 'bottom', 'pad':0.25,'shrink': 0.6},
                           ax = ax)

                ax.set_title(MM)
                ax.set_ylabel("Target sequence")

            fig.suptitle(f"IFC barcode: {self.experiment_name}") 
        
        else :

            fig, ax = plt.subplots(1,1, figsize = (18,8))
            
            MM = master_mixes[0] 
                
            data = self.result.query('master_mix == @MM').pivot(columns = "primer_set", index = ["target_seq","target_conc"], values = 'proportion_reps_amped')

            sns.heatmap(data =data, square = True, cmap = "binary",
                        linewidth = 0.1, linecolor = 'darkgrey',vmin =0, vmax = 1,
                       cbar_kws = {'label':f'Proportion reps amplified',
                                   'location' : 'bottom', 'pad':0.25,'shrink': 0.6},
                       ax = ax)

            ax.set_title(MM)
            ax.set_ylabel("Target sequence")

            fig.suptitle(f"IFC barcode: {self.experiment_name}") 
        
        if savefig == True : 
            
            fig.savefig(f"{self.figure_output_path}/heatmap_proportion_reps_amped.pdf", format = "pdf", bbox_inches = 'tight')
            fig.savefig(f"{self.figure_output_path}/heatmap_proportion_reps_amped.png", format = "png", bbox_inches = 'tight')
            
            
    def plot_all_targs_one_pset(self, pset_to_plot, savefig = False):

        fig, axes = plt.subplots(2,1, figsize = (10,6))
        fig.tight_layout(h_pad = 8)

        master_mixes = sorted(list({value.get('master_mix') for key, value in self.annotation_dict.items()}))

        for MM, ax in zip(master_mixes,axes.ravel()) : 
            data = self.chamber_df.query('master_mix == @MM & primer_set == @pset_to_plot').sort_values(by = "target_seq")

            sns.stripplot(data =data, x = 'target_seq', y = 'tp',
                       ax = ax)

            plot_title = f"Tp for amplification of all targets by {pset_to_plot}, {MM}"

            ax.set_title(plot_title)
            ax.set_ylabel("Time to positive (min)")
            ax.set_xlabel("Target sequence")
            ax.set_xticks(ticks = ax.get_xticks(), labels = ax.get_xticklabels(), rotation = 45, ha = 'right');

        fig.suptitle(f"IFC barcode: {self.experiment_name}", y = 1.1) 
        
        if savefig == True : 
            fig.savefig(f"{self.figure_output_path}/{pset_to_plot}_tp_stripplot.pdf", format = "pdf", bbox_inches = 'tight')
            fig.savefig(f"{self.figure_output_path}/{pset_to_plot}_tp_stripplot.png", format = "png", bbox_inches = 'tight')
            
    def plot_all_psets_one_target(self, targ_to_plot, savefig = False):

        fig, axes = plt.subplots(2,1, figsize = (10,6))
        fig.tight_layout(h_pad = 8)

        master_mixes = sorted(list({value.get('master_mix') for key, value in self.annotation_dict.items()}))

        for MM, ax in zip(master_mixes,axes.ravel()) : 
            data = self.chamber_df.query('master_mix == @MM & target_seq == @targ_to_plot').sort_values(by = "primer_set")

            sns.stripplot(data =data, x = 'primer_set', y = 'tp',
                       ax = ax)

            plot_title = f"Tp for amplification of {targ_to_plot} by all primer sets, {MM}"

            ax.set_title(plot_title)
            ax.set_ylabel("Time to positive (min)")
            ax.set_xlabel("Primer set")
            ax.set_xticks(ticks = ax.get_xticks(), labels = ax.get_xticklabels(), rotation = 45, ha = 'right');

        fig.suptitle(f"IFC barcode: {self.experiment_name}", y = 1.1) 
        
        if savefig == True : 
            fig.savefig(f"{self.figure_output_path}/{targ_to_plot}_tp_stripplot.pdf", format = "pdf", bbox_inches = 'tight')
            fig.savefig(f"{self.figure_output_path}/{targ_to_plot}_tp_stripplot.png", format = "png", bbox_inches = 'tight')


