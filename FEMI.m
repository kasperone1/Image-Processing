close all
clear variables

raw_image_listing=dir('*Z0*.tiff');
filename=raw_image_listing(1).name;
filename = filename(1:end-8);
fname_channel_PSELECTIN = [filename '_C0.tiff'];   %PSELECTIN 488
fname_channel_CD41 = [filename '_C1.tiff'];  %CD41 561
fname_channel_FIBRIN = [filename '_C2.tiff'];  %FIBRIN
fname_channel_OPEN = [filename '_C3.tiff'];  %OPEN


%read the log file
log_filename = dir('*.log');
log_fileID = fopen(log_filename.name);
string=textscan(log_fileID,'%s %s %s %s','HeaderLines',1,'CollectOutput',1);
fclose(log_fileID);
log_text_line1 = char(string{1,1}(1,:));

%read capture date
capture_time = [log_text_line1(1,:) log_text_line1(2,:) ' ' ...
    log_text_line1(3,:) log_text_line1(4,:)];

%read resolution
log_text_line6 = char(string{1,1}(5,:));
pixel_resolution_microns_per_pixel = str2double(log_text_line6(4,:)); %microns per pixel

time_resolution_seconds_interval = 1; %seconds per interval

platelet_diameter_microns = 1 %microns
platelet_diameter_pixels = platelet_diameter_microns/pixel_resolution_microns_per_pixel;
platelet_radius_pixels = platelet_diameter_pixels / 2;
platelet_area_pixels = round(pi *platelet_radius_pixels ^ 2);

platelet_diameter_min_microns = platelet_diameter_microns - 0.5 %microns
platelet_diameter_max_microns = platelet_diameter_microns + 0.2 %microns

platelet_diameter_min_pixels = round(platelet_diameter_min_microns/pixel_resolution_microns_per_pixel);
platelet_diameter_max_pixels = round(platelet_diameter_max_microns/pixel_resolution_microns_per_pixel);

platelet_radius_min_pixels = platelet_diameter_min_pixels / 2;
platelet_radius_max_pixels = platelet_diameter_max_pixels / 2;

platelet_area_min_pixels = round(pi * platelet_radius_min_pixels ^ 2);
platelet_area_max_pixels = round(pi * platelet_radius_max_pixels ^ 2);

bw_area_cutoff = 2*platelet_area_min_pixels;

bw_area_cutoff_PSELECTIN = platelet_area_max_pixels;

info = imfinfo(fname_channel_CD41);
num_images = numel(info);

image_height = info(1).Height;
image_width = info(1).Width;

blood_vessel_diameter_pixels = 130; %user input

lower_bound = 15;
% lower_bound = 100;
upper_bound = num_images;
% upper_bound = 100;


% the current IFD.
% Open a TIFF file that contains images and subimages using the Tiff object constructor. This example uses the TIFF file created in Creating Subdirectories in a TIFF File, which contains one IFD directory with two subIFDs. The Tiff constructor opens the TIFF file, and makes the first subIFD in the file the current IFD:
t = Tiff(fname_channel_OPEN,'r');
% % Retrieve the locations of subIFDs associated with the current IFD. Use the getTag method to get the value of the SubIFD tag. This returns an array of byte offsets that specify the location of subIFDs:
% offsets = t.getTag('SubIFD')
% % Navigate to the first subIFD using the setSubDirectory method. Specify the byte offset of the subIFD as an argument. This call makes the subIFD the current IFD:
% t.setSubDirectory(offsets(1));
% % Read the image data from the current IFD (the first subIFD) as you would with any other IFD in the file:
% subimage_one = t.read();
% % View the first subimage:
% imagesc(subimage_one)
% % To view the second subimage, call the setSubDirectory method again, specifying the byte offset of the second subIFD:
% t.setSubDirectory(info(2).Offset);
% % Read the image data from the current IFD (the second subIFD) as you would with any other IFD in the file:
% subimage_two = t.read();
% % View the second subimage:
% % imagesc(subimage_two)
% % Close the Tiff object.
% % t.close();
% % t_in.getTag(t_in.TagID.ImageWidth)

number_images_analyzed = upper_bound - lower_bound + 1;
channel_OPEN_ALL_slices = zeros(number_images_analyzed,image_height,image_width);

for frame_number = lower_bound:upper_bound
    t.setDirectory(frame_number);
    channel_OPEN_this_slice = t.read();
    channel_OPEN_ALL_slices(frame_number,:,:) = channel_OPEN_this_slice;
    
end %frame_number = lower_bound:upper_bound

%% MASK OUT PIXELS THAT DO NOT VARY MUCH OVER THE COURSE OF THE FLASH
% [BW_ALBUMIN_THIS_FLASH_REDUCED_stdev_mask] = ...
%     make_variability_mask( ...
%     channel_OPEN_ALL_slices);
% meanbk=squeeze(mean(channel_OPEN_ALL_slices,1));

platelet_inter_distances_avg = nan(number_images_analyzed,1);
platelet_inter_distances_std = nan(number_images_analyzed,1);
nearest_neighbor_distance_avg = nan(number_images_analyzed,1);
nearest_neighbor_distance_std = nan(number_images_analyzed,1);

%% DEFINE RGB
red = cat(3,ones(image_height,image_width),zeros(image_height,image_width),zeros(image_height,image_width));
green = cat(3,zeros(image_height,image_width),ones(image_height,image_width),zeros(image_height,image_width));
blue = cat(3,zeros(image_height,image_width),zeros(image_height,image_width),ones(image_height,image_width));

yellow = cat(3,ones(image_height,image_width),ones(image_height,image_width),zeros(image_height,image_width));


BLOOD_VESSEL_BOTH_WALLS_MASK_filename = 'BLOOD_VESSEL_BOTH_WALLS_MASK.bmp';
BLOOD_VESSEL_WALL_BOTTOM_MASK_filename = 'BLOOD_VESSEL_WALL_BOTTOM_MASK.bmp';
BLOOD_VESSEL_WALL_TOP_MASK_filename = 'BLOOD_VESSEL_WALL_TOP_MASK.bmp';
BLOOD_VESSEL_BOTH_WALLS_BORDER_MASK_filename = 'BLOOD_VESSEL_BOTH_WALLS_BORDER_MASK.bmp';

rotation_angle_file = 'rotation_angle';

if(exist(BLOOD_VESSEL_BOTH_WALLS_MASK_filename) && rotation_angle_file) %read the blood vessel walls from existing mask
    
    imread(BLOOD_VESSEL_BOTH_WALLS_MASK_filename)
    
    %read the rotation angle from rotatation angle file
    rotation_angle;
    
else %ask the user to draw the blood vessel walls\
    
    %pick the rotation angle
    [ rotation_angle ] = rotate_image( channel_OPEN_ALL_slices );
    
    % %    [ BLOOD_VESSEL_BOTH_WALLS_MASK,...
    % %      BLOOD_VESSEL_WALL_TOP_MASK,...
    % %      BLOOD_VESSEL_WALL_BOTTOM_MASK,...
    % %      BLOOD_VESSEL_BOTH_WALLS_BORDER_MASK] = locate_blood_vessels_by_user( ...
    % %      channel_OPEN_ALL_slices,...
    % %      BLOOD_VESSEL_BOTH_WALLS_MASK_filename,...
    % %      BLOOD_VESSEL_WALL_BOTTOM_MASK_filename,...
    % %      BLOOD_VESSEL_WALL_TOP_MASK_filename,...
    % %      BLOOD_VESSEL_BOTH_WALLS_BORDER_MASK_filename,...
    % %      rotation_angle);
end


channel_CD41_all_time_steps = uint16(zeros(number_images_analyzed,image_height,image_width));
bw_CD41_mask_ALL_slices = false(number_images_analyzed,image_height,image_width);
BW=false(number_images_analyzed,image_height,image_width);

obj = setupSystemObjects(platelet_area_min_pixels);

tracks = initializeTracks(); % Create an empty array of tracks.

nextId = 1; % ID of the next track

frame_counter = 0;

mask_file_names = dir('*png');
 
total_png_files = length(mask_file_names);

for frame_number = lower_bound:upper_bound
    
    fprintf(1,'reading frame_number = %i \n',frame_number);
    
    %% READ DATA
    channel_PSELECTIN_this_slice = imread(fname_channel_PSELECTIN, frame_number, 'Info', info);
    channel_CD41_this_slice = imread(fname_channel_CD41, frame_number, 'Info', info);
    channel_FIBRIN_this_slice = imread(fname_channel_FIBRIN, frame_number, 'Info', info);
    channel_OPEN_this_slice = uint16(squeeze(channel_OPEN_ALL_slices(frame_number,:,:)));
%   rotate all slices
%   channel_PSELECTIN_this_slice = imrotate(channel_PSELECTIN_this_slice,rotation_angle,'crop');
%   channel_CD41_this_slice = imrotate(channel_CD41_this_slice,rotation_angle,'crop');
%   channel_FIBRIN_this_slice = imrotate(channel_FIBRIN_this_slice,rotation_angle,'crop');
%   channel_OPEN_this_slice = imrotate(channel_OPEN_this_slice,rotation_angle,'crop');
    
    if(frame_number == lower_bound)
        
        channel_OPEN_background = channel_OPEN_this_slice ;
        
    end %frame_number == lower_bound
    
    %% MASK CHANNELS
    %PESELECTIN
    seg_I = imquantize(channel_PSELECTIN_this_slice,multithresh(channel_PSELECTIN_this_slice,2));
    bw_PSELECTIN_mask_this_slice = seg_I==3;
    I=channel_PSELECTIN_this_slice;
    I(~bw_PSELECTIN_mask_this_slice)=0;
    seg_I2 = imquantize(I,multithresh(I,2));
    bw_PSELECTIN_mask_this_slice = seg_I2==3;
    bw_PSELECTIN_mask_this_slice = bwmorph(bw_PSELECTIN_mask_this_slice,'spur');
    bw_PSELECTIN_mask_this_slice = bwmorph(bw_PSELECTIN_mask_this_slice,'clean');
    bw_PSELECTIN_mask_this_slice = bwareaopen(bw_PSELECTIN_mask_this_slice,bw_area_cutoff*2);
    bw_PSELECTIN_mask_this_slice = bwmorph(bw_PSELECTIN_mask_this_slice,'fill');
    
    %     figure(3),imshowpair(bw_PSELECTIN_mask_this_slice,channel_PSELECTIN_this_slice)
    %     pause(0.5)
    %     close(3)
    
    %FIBRIN
    seg_I = imquantize(channel_FIBRIN_this_slice,multithresh(channel_FIBRIN_this_slice,2));
    bw_FIBRIN_mask_this_slice = seg_I==3;
    bw_FIBRIN_mask_this_slice = bwmorph(bw_FIBRIN_mask_this_slice,'spur');
    bw_FIBRIN_mask_this_slice = bwmorph(bw_FIBRIN_mask_this_slice,'clean');
    bw_FIBRIN_mask_this_slice = bwareaopen(bw_FIBRIN_mask_this_slice,bw_area_cutoff*2);
    bw_FIBRIN_mask_this_slice = bwmorph(bw_FIBRIN_mask_this_slice,'fill');
    %     bw_FIBRIN_mask_this_slice=im2bw(channel_FIBRIN_this_slice,graythresh(channel_FIBRIN_this_slice));
    %     bw_FIBRIN_mask_this_slice = bwmorph(bw_FIBRIN_mask_this_slice,'fill');
    %     bw_FIBRIN_mask_this_slice = bwmorph(bw_FIBRIN_mask_this_slice,'spur');
    %     bw_FIBRIN_mask_this_slice = bwmorph(bw_FIBRIN_mask_this_slice,'clean');
    %     imshowpair(bw_FIBRIN_mask_this_slice,channel_FIBRIN_this_slice)
    %     imshowpair(channel_CD41_this_slice,bw_PSELECTIN_mask_this_slice)
    %% SEGMENT PLATELETS
    % [bw_CD41_mask_this_slice] = segment_platelet_tracers( ...
    %     channel_CD41_this_slice,...
    %     platelet_diameter_max_pixels,...
    %     frame_number);
    %         [ bw_CD41_mask_this_slice] = ...
    %             segmentation_new(...
    %             channel_CD41_this_slice,...
    %             frame_number);
    %     %     bw_CD41_mask_this_slice=im2bw(channel_CD41_this_slice,graythresh(channel_CD41_this_slice));
    %     imshowpair(bw_CD41_mask_this_slice,channel_CD41_this_slice)
    %     title(['frame_number = ' num2str(frame_number) '; LEFT = MASK,  RIGHT = ORIGINAL']);
    %     pause(0.1)
   kmeans_thresh = multithresh(channel_CD41_this_slice,2);
   seg_I = imquantize(channel_CD41_this_slice,kmeans_thresh);
   level_of_interest = 3;
   seg_I(seg_I~=level_of_interest)=false;
   seg_I(seg_I==level_of_interest)=true;
%     seg_I = imopen(seg_I,strel('disk',4));
%    multithresh_bw_mask = seg_I;
   BW_filled_holes = imfill(seg_I,'holes');
%    imshowpair(channel_CD41_this_slice,label2rgb(seg_I),'montage');
%  b=imadjust(channel_CD41_this_slice);
   b = medfilt2(channel_CD41_this_slice,[3 3]);
   b = imtophat(channel_CD41_this_slice,strel('disk',5));    
   A = adapthisteq(channel_CD41_this_slice,'clipLimit',0.01,...
        'Distribution','rayleigh');
   kmeans_thresh = multithresh(A,2);
   seg_I = imquantize(A,kmeans_thresh);
   level_of_interest = 3;
   seg_I(seg_I~=level_of_interest)=false;
   seg_I(seg_I==level_of_interest)=true;
   seg_I = imopen(seg_I,strel('disk',4));
% imshowpair(channel_CD41_this_slice, seg_I,'montage');
% imshowpair(channel_CD41_this_slice,uint16(BW_filled_holes),'montage');
   mean_intensity(frame_number) = mean(channel_CD41_this_slice(:));
   mean_intensity_allimages=mean(mean_intensity(:));
   channel_CD41_this_slice_1= imadjust(channel_CD41_this_slice,[0.01 0.99],[]);
%  imshowpair(channel_CD41_this_slice,channel_CD41_this_slice_1,'montage')
   channel_CD41_this_slice=channel_CD41_this_slice_1;
     [ bw_CD41_mask_this_slice] = ...
         segmentation_watershed(...
         channel_CD41_this_slice,...
         frame_number); 
    bw4_perim = bwperim(bw_CD41_mask_this_slice);
%   overlay1 = imoverlay(channel_CD41_this_slice, bw4_perim, [.3 1 .3]); 
    overlay1 = channel_CD41_this_slice;
    overlay1(bw4_perim) = 65536;
    mask_em = imextendedmax(overlay1, 30);
    mask_em = imopen(mask_em,strel('arbitrary',[0 0 0;0 1 0;0 0 0]));
    mask_em= bwareaopen(mask_em,16);
    mask_em=mask_em~=0;
    mask_em=~mask_em;
    mask_em_L = bwlabel(mask_em,4);
    mask_em(mask_em_L==1)=0; 
%   stats = regionprops(mask_em_L,'Area')
%   areas = [stats.Area]
%   [maxArea largestBlobIndex] = max(area)

    if(frame_number~=57 & frame_number~=60 & frame_number~=61 & frame_number~=78 & frame_number~=84 & frame_number(:)~=86 & frame_number~=92);
        [bw_CD41_mask_this_slice] =segmentation_watershed(channel_CD41_this_slice,frame_number);
    else
        [~,kmeans_mask]=kmeans_ima(channel_CD41_this_slice,3);%clustering the image into three parts
        kmeans_thresh = multithresh(kmeans_mask,2);
        seg_mas = imquantize(kmeans_mask,kmeans_thresh);
        level_of_interest = 3;
        seg_mas(seg_mas~=level_of_interest)=false;
        seg_mas(seg_mas==level_of_interest)=true;
        bw_CD41_mask_this_slice=logical(seg_mas);
    end 
%     imwrite(bw_CD41_mask_this_slice,['mask_current _ ' num2str(frame_number)],'png');
%     movefile(['mask_current _ ' num2str(frame_number) ], sprintf('mask_%03d.png', frame_number));
%     mask_current =imread(['mask_current _ ' num2str(frame_number)],'png')   
%      for ii=1:total_png_files
%         mask_current_name = mask_file_names(ii).name;
%         current_image = imread(mask_current_name);
%         frame = channel_CD41_this_slice;
%         mask = current_image;
%         [~,centroids, bboxes] = obj.blobAnalyser.step(mask);
%      end
%   imshowpair(bw_CD41_mask_this_slice,channel_CD41_this_slice,'montage')
%   title(['frame_number = ' num2str(frame_number) '; LEFT = MASK,  RIGHT = ORIGINAL'])
%   pause(0.2)
%     [mask1,mu,v,p]=EMSeg(channel_CD41_this_slice,3);
%     kmeans_thresh = multithresh(mask1,2);
%     seg_mask1 = imquantize(mask1,kmeans_thresh);
%     level_of_interest = 3;
%     seg_mask1(seg_mask1~=level_of_interest)=false;
%     seg_mask1(seg_mask1==level_of_interest)=true;
%     imshowpair(seg_mask1,channel_CD41_this_slice,'montage')
%     title(['frame_number = ' num2str(frame_number) '; LEFT = MASK,  RIGHT = ORIGINAL'])
%     pause(0.2)
%     bw_CD41_mask_this_slice=im2bw(channel_CD41_this_slice,graythresh(channel_CD41_this_slice));  
    %% GET PLATELET POSITIONS FROM PLATELET MASK
    if(frame_number == lower_bound)
        particle_position_list(1) = 0;
        object_counter = 0;
    end %frame_number == lower_bound
%     [bw_foreground_weightedcentroids_mask,...
%         particle_position_list,...
%         object_counter] = ...
%         find_object_centroids(...
%         channel_CD41_this_slice,...
%         bw_CD41_mask_this_slice,...
%         frame_number,...
%         particle_position_list,...
%         object_counter);
    frame = channel_CD41_this_slice;
    mask = bw_CD41_mask_this_slice;

    [~,centroids, bboxes] = obj.blobAnalyser.step(mask);
    
%   mask = imopen(mask, strel('disk',2));
%   mask = imclose(mask, strel('disk',2));
%   mask = imopen(mask, strel('diamond',2));
%   [centroids2, bboxes2, mask2] = detectObjects(mask,obj);
%   imshowpair(frame,mask)
%   title(num2str(frame_number))
    %CALCULATE INTERPLATELET DISTANCES
    %     [platelet_inter_distances_avg,...
    %         platelet_inter_distances_std,...
    %         nearest_neighbor_distance_avg,...
    %         nearest_neighbor_distance_std] = ...
    %         inter_platelet_distances(...
    %         bw_CD41_mask_this_slice,...
    %         channel_CD41_this_slice,...
    %         platelet_inter_distances_avg,...
    %         platelet_inter_distances_std,...
    %         nearest_neighbor_distance_avg,...
    %         nearest_neighbor_distance_std,...
    %         lower_bound,...
    %         frame_number);
    %     subplot(1,2,2),plot(lower_bound:frame_number,platelet_mean_separation(lower_bound:frame_number))
    %     imshow(bw_PSELECTIN_mask_this_slice)
    %     pause(0.1)
    %% FINAL OVERLAY
    %     channel_CD41_this_slice(~bw_CD41_mask_this_slice)=0;
    %     channel_FIBRIN_this_slice(~bw_FIBRIN_mask_this_slice)=0;
    %     channel_PSELECTIN_this_slice(~bw_PSELECTIN_mask_this_slice)=0;
    % %     channel_PSELECTIN_this_slice(~bw_FIBRIN_mask_this_slice)=0;
    %
    %     brighten_const=10;
    %     red_influence_map = (channel_CD41_this_slice*brighten_const);
    %     green_influence_map = (channel_FIBRIN_this_slice*brighten_const);
    %     blue_influence_map = (channel_PSELECTIN_this_slice*brighten_const);
    %
    %     yellow_influence_map = (channel_PSELECTIN_this_slice*brighten_const);
    %
    %     imshow(channel_OPEN_this_slice)
    %     hold on
    %     h_red = imshow(red);
    %     h_green = imshow(green);
    %     h_blue = imshow(blue);
    %     h_yellow = imshow(yellow);
    %
    %     hold off
    %     set(h_red,'AlphaData',red_influence_map)
    %     set(h_green,'AlphaData',green_influence_map)
    %     set(h_blue,'AlphaData',blue_influence_map)
    %     set(h_yellow,'AlphaData',yellow_influence_map)
    %
    %     hold off
%         pause(0.1)
    %     close
    
    %% tracking
    % Solve the assignment problem.
    costOfNonAssignment = 8;
    invisibleForTooLong = 20;  %total number of invisible (not necessarily consecutively) frames 
    ageThreshold = 18;  %trajectory must be last at least this long to keep
    Karman_Filter_Motion_Model = 'ConstantVelocity';
%      Karman_Filter_Initial_Estimate_Error = [200, 50];
%      Karman_Filter_Initial_Estimate_MotionNoise = [100, 25];
%      Karman_Filter_Initial_Estimate_MeasurementNoise = 100;

% Initial estimate uncertainty variance, specified as a two- or three-element vector. 
% The initial estimate error specifies the variance of the initial estimates of location, velocity, and acceleration of the tracked object. 
% The function assumes a zero initial velocity and acceleration for the object, at the location you set with the InitialLocation property. 
% You can set the InitialEstimateError to an approximated value: (assumed values – actual values)2 + the variance of the values
% The value of this property affects the Kalman filter for the first few detections. 
% Later, the estimate error is determined by the noise and input data. 
% A larger value for the initial estimate error helps the Kalman filter to adapt to the detection results faster. 
% However, a larger value also prevents the Kalman filter from removing noise from the first few detections.
      Karman_Filter_Initial_Estimate_Error.LocationVariance = [234, 202];  %[LocationVariance, VelocityVariance]

%       Deviation of selected and actual model, specified as a two- or three-element vector. 
%       The motion noise specifies the tolerance of the Kalman filter for the deviation from the chosen model. 
%       This tolerance compensates for the difference between the object's actual motion and that of the model you choose.
%       Increasing this value may cause the Kalman filter to change its state to fit the detections. 
%       Such an increase may prevent the Kalman filter from removing enough noise from the detections. 
%       The values of this property stay constant and therefore may affect the long-term performance of the Kalman filter.
      
      Karman_Filter_Initial_Estimate_Error.VelocityVariance = [100, 25];  %[LocationVariance, VelocityVariance]
      
%       Variance inaccuracy of detected location, specified as a scalar. 
% It is directly related to the technique used to detect the physical objects. 
% Increasing the MeasurementNoise value enables the Kalman filter to remove more noise from the detections. 
% However, it may also cause the Kalman filter to adhere too closely to the motion model you chose, 
% putting less emphasis on the detections. The values of this property stay constant, 
% and therefore may affect the long-term performance of the Kalman filter.

      Karman_Filter_Initial_Estimate_Error.AccelerationVariance = 150;
    
    [tracks] = predictNewLocationsOfTracks(tracks);
    
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment(tracks,centroids,costOfNonAssignment);
    
    [tracks] = updateAssignedTracks(assignments,centroids,bboxes,tracks);
    
    [tracks] = updateUnassignedTracks(tracks,unassignedTracks);
    
    [tracks] = deleteLostTracks(tracks,invisibleForTooLong,ageThreshold);
    
    [tracks,nextId] = createNewTracks(tracks,centroids,bboxes,...
        Karman_Filter_Motion_Model,...
        Karman_Filter_Initial_Estimate_Error,...
        unassignedDetections,nextId);
    
      displayTrackingResults(frame,mask,tracks,obj);  
%       pause 
      
    %% saving arrays 
    channel_CD41_all_time_steps(frame_number,:,:) = channel_CD41_this_slice;
    bw_CD41_mask_ALL_slices(frame_number,:,:) = bw_CD41_mask_this_slice; 
end %
% Adjusting the image histogram for failing frames
frame_all = (lower_bound:upper_bound);
a=zeros(456,508);
% counter = 0;
f=zeros(1,length(frame_all));
for j = lower_bound:upper_bound
%   if(frame_number~=57||frame_number~=60||frame_number~=61||frame_number~=78||frame_number~=84||frame_number~=86||frame_number~=92);
    f=frame_all(frame_all(:)~=57 & frame_all(:)~=60 & frame_all(:)~=61 & frame_all(:)~=78 & frame_all(:)~=84 & frame_all(:)~=86 & frame_all(:)~=92);
%     counter = counter + 1;   
end
mean_individual=zeros(1,length(f));
for j = f(1):f(end)
       channel_CD41_this_slice = imread(fname_channel_CD41, j, 'Info', info);
       a=double(a);
       a=a+double(channel_CD41_this_slice);
       mean_individual(j)=mean(channel_CD41_this_slice(:));
       mean_goodframes=mean(channel_CD41_this_slice(:));
end
a=a/length(f);
a=uint16(a);
count=0;
% x=[f(1):f(end)];
% y=mean_individual([15:length(mean_individual)]);
for frame_number=[57,60,61,78,84,86,92]
    count=count+1;
    channel_CD41_this_slice = imread(fname_channel_CD41, frame_number, 'Info', info);
    B = imhistmatch(channel_CD41_this_slice,a);
%   imshowpair(B,channel_CD41_this_slice,'montage')
%   pause(0.2)     
end

mean_values=zeros(1,count);
adjusted_mean=zeros(1,count);
frame_number=[57,60,61,78,84,86,92];
for k=1:count
    channel_CD41_this_slice = imread(fname_channel_CD41, frame_number(k), 'Info', info);   
    mean_values(k)=mean(channel_CD41_this_slice(:));
    mean_badframes=mean(mean_values);
    B = imhistmatch(channel_CD41_this_slice,a);
    adjusted_mean(k)=mean(B(:));
%     imshowpair(B,channel_CD41_this_slice,'montage')
%     pause(0.2) 
end
% k=1:count;
% plot(k,mean_values,k,adjusted_mean)
% l=mean(adjusted_mean(:));
% l1=mean(mean_values(:));
% diff=abs(l-l1);
stop
% TRACE PLATELET TRAJECTORIES
param.mem = 3; %number of frames that a particle can be lost
param.good = 5; %trajectory must be last at least this long to keep
param.dim = length(particle_position_list(1,:))-1; %no idea
param.quiet = 0; %suppress text output

particle_trajectories = ...
    track(...
    particle_position_list,...
    platelet_diameter_max_microns*5,...
    param);

total_number_of_particles = max(particle_trajectories(:,4));
total_number_of_time_steps = upper_bound;

%sort tracking data by time
particle_trajectories_time_sorted = sortrows(particle_trajectories,3);

% INTERPOLATE MISSING PLATET POSITIONS
[ all_particle_time_steps,...
    all_particle_positions_x,...
    all_particle_positions_y,...
    all_particle_time_steps_min,...
    all_particle_time_steps_max...
    ] = ...
    interp_nan_trajectories(...
    total_number_of_particles,...
    upper_bound,...
    particle_trajectories);

% SMOOTH PARTICLE TRAJECTORIES
[ all_particle_positions_x ] = smooth_particle_trajectories( ...
    all_particle_positions_x,...
    all_particle_time_steps_min,...
    all_particle_time_steps_max,...
    total_number_of_particles);

[ all_particle_positions_y ] = smooth_particle_trajectories( ...
    all_particle_positions_y,...
    all_particle_time_steps_min,...
    all_particle_time_steps_max,...
    total_number_of_particles);


particle_colormap = rand(total_number_of_particles,3);
colormap(particle_colormap);

%% LABEL PLATELETS BY TRACKED OBJECT NUMBER
for frame_number = lower_bound:upper_bound
    
    channel_CD41_this_slice = squeeze(channel_CD41_all_time_steps(frame_number,:,:));
    bw_CD41_mask_this_slice = squeeze(bw_CD41_mask_ALL_slices(frame_number,:,:));
    
    % cc = bwconncomp(bw_CD41_mask_this_slice, 4);
    % platelet_single = false(size(bw_CD41_mask_this_slice));
    % platelet_single(cc.Centroid{particle_trajectories(:,4)}) = true;
    % imshow(label2rgb(platelet_single,'jet','w'));
    % pause(0.1)
    %assign random color to each particle
    
    %determine which objects from current data exist in the current frame
    current_frame_objects_rows = particle_trajectories_time_sorted(:,3)==frame_number;
    
    total_number_of_objects_current_frame = nnz(current_frame_objects_rows);
    
    %fetch objects that exist in current frame from tracking data
    current_frame_objects_all_data = particle_trajectories_time_sorted(current_frame_objects_rows,:);
    
    %make a structuring element with object location in the current frame
    %and corresponding object number
    current_frame_objects.Location = current_frame_objects_all_data(:,1:2);
    current_frame_objects.ObjectLabel = current_frame_objects_all_data(:,4);
    
    %MAKE A BW POSITION MASK FOR OBJECTS THAT EXIST IN THIS FRAME
    LABEL_MATRIX_ALL_Objects_This_Frame = zeros(size(bw_CD41_mask_this_slice));
    
    for object_number = 1:total_number_of_objects_current_frame
        
        %make a blank matrix for every new object
        bw_current_fram_objects_locations_MASK = false(size(bw_CD41_mask_this_slice));
        
        %make the centroid marker bw mask
        bw_current_fram_objects_locations_MASK(current_frame_objects.Location(object_number,2),...
            current_frame_objects.Location(object_number,1))=true;
        
        %use the marker mask to pick out the marked object from the rest of
        %the objects
        bw_CD41_mask_THIS_OBJECT_this_slice = ...
            imreconstruct(bw_current_fram_objects_locations_MASK,bw_CD41_mask_this_slice);
        
        %label the chosen object using the marker's label #
        LABEL_MATRIX_ALL_Objects_This_Frame(bw_CD41_mask_THIS_OBJECT_this_slice) = ...
            current_frame_objects.ObjectLabel(object_number);
        
        
        %         bw_current_fram_objects_locations_LABEL_MATRIX(current_frame_objects.Location(object_number,2),...
        %             current_frame_objects.Location(object_number,1))=current_frame_objects.ObjectLabel(object_number);
        
    end %object_number = 1:total_number_of_objects_current_frame
    
%     imshowpair(label2rgb(LABEL_MATRIX_ALL_Objects_This_Frame,...
%         particle_colormap,'k','noshuffle'),channel_CD41_this_slice,'montage');
%     title(['Frame Number = ' num2str(frame_number)])
%     
%     pause(1)
    % [BW_ALL_OBJECTS_Label_Matrix, total_number_of_unique_objects] = ...
    %     bwlabel(bw_CD41_mask_this_slice);
end % frame_number = lower_bound:upper_bound

%declaring variables for statistics
Distance_2PSELECTIN = bwdist(bw_PSELECTIN_mask_this_slice);

all_steps_velocity_magnitude_mean=nan(upper_bound,1);
all_steps_Distance_2PSELECTIN_mean=nan(upper_bound,1);
particle_counter_all_frames=nan(upper_bound,1);

for frame_number = lower_bound:upper_bound
    
    overlay_this_time_step = squeeze(channel_CD41_all_time_steps(frame_number,:,:));
    overlay_this_time_step = perimeter_imoverlay(uint16(overlay_this_time_step),...
        bw_PSELECTIN_mask_this_slice, [1 0 1]);
    imshow(overlay_this_time_step)
    
    hold on
    
    particle_counter_this_frame = 0;
    this_step_velocity_magnitude_sum = 0;
    Distance_2PSELECTIN_this_step_sum = 0;
    
    for this_particle_number = 1:1
        
        scatter(...
            all_particle_positions_x(this_particle_number,lower_bound:frame_number),...
            all_particle_positions_y(this_particle_number,lower_bound:frame_number),...
            20,...
            particle_colormap(this_particle_number,:) )
        %
        %         if(frame_number>lower_bound)
        %
        %             this_particle_position_THIS_time_step_x = all_particle_positions_x(this_particle_number,frame_number);
        %             this_particle_position_LAST_time_step_x = all_particle_positions_x(this_particle_number,frame_number-1);
        %
        %             this_particle_position_THIS_time_step_y = all_particle_positions_y(this_particle_number,frame_number);
        %             this_particle_position_LAST_time_step_y = all_particle_positions_y(this_particle_number,frame_number-1);
        %
        %             this_particle_velocity_THIS_time_step_x = ...
        %                 this_particle_position_THIS_time_step_x - this_particle_position_LAST_time_step_x;
        %
        %             this_particle_velocity_THIS_time_step_y = ...
        %                 this_particle_position_THIS_time_step_y - this_particle_position_LAST_time_step_y;
        %
        %             this_particle_velocity_magnitude_time_step = sqrt(this_particle_velocity_THIS_time_step_x^2 +...
        %                 this_particle_velocity_THIS_time_step_y^2);
        %
        %             if(isfinite(this_particle_velocity_magnitude_time_step))
        %
        %                 particle_counter_this_frame = particle_counter_this_frame + 1;
        %                 this_step_velocity_magnitude_sum = this_step_velocity_magnitude_sum + ...
        %                     this_particle_velocity_magnitude_time_step;
        %
        %                 Distance_2PSELECTIN_this_particle_this_step = ...
        %                     interp2(Distance_2PSELECTIN,...
        %                     this_particle_position_THIS_time_step_x,...
        %                     this_particle_position_THIS_time_step_y);
        %
        %                 Distance_2PSELECTIN_this_step_sum = Distance_2PSELECTIN_this_step_sum + ...
        %                     Distance_2PSELECTIN_this_particle_this_step;
        %
        %             end %isfinite(this_step_velocity_magnitude_sum)
        %
        %         end %frame_number>lower_bound
        
    end % this_particle_number = 1:total_number_of_particles
    
    hold off
    pause(0.1)
    
    particle_counter_all_frames(frame_number) = particle_counter_this_frame;
    
    all_steps_velocity_magnitude_mean(frame_number) = ...
        this_step_velocity_magnitude_sum / particle_counter_this_frame * ...
        pixel_resolution_microns_per_pixel / time_resolution_seconds_interval;
    
    all_steps_Distance_2PSELECTIN_mean(frame_number) = ...
        (Distance_2PSELECTIN_this_step_sum / particle_counter_this_frame)*pixel_resolution_microns_per_pixel;
    
end %frame_number = lower_bound:upper_bound

figure,plot(all_steps_velocity_magnitude_mean)
figure,plot(all_steps_Distance_2PSELECTIN_mean)