clear
close all
format long
delete '*.tiff';
dir_loc = ...
'/home/phemykadri/Desktop/Scaffold_files/all_files/fiber scaffolds.zip (Unzipped Files)/Background_removed';
% optional.
dir_loc2 = '/home/phemykadri/Desktop/Scaffold_files/all_files/fiber scaffolds.zip (Unzipped Files)/fiber_fine';
img_format = '*.tif';%specify image format.Image is either in dcm,tif or tiff formats.
%optional.
img_format2 = '*.dcm';
%determine operating system
operating_system = computer;
if( strcmpi(operating_system,'GLNXA64') ) 
    os_slash ='/';%slash character for linux. 
else
    os_slash ='\';%slash character for windows. 
end;
dim = strcat(dir_loc,os_slash,img_format);%extract image files only.
%optional.
dim_2 = strcat(dir_loc2,os_slash,img_format2);
A = dir(dim);
%optional.
A2 = dir(dim_2);
white_intensity_value = 2^8 -1;%white intensity value for 8bit image
slice = zeros(length(A),length(A),length(A));
slice_original = zeros(size(slice));
start = 1;
finish = length(A);
%cropping rectangle.Determined by obtaining rectangle containing region of
%interest in brightest slice of the stack.
crop_rect = [125,104,266,256];%determined by trial and error.
begin = 10;
stop = 380;
for image_num = start:finish 
    X2 = dicomread(strcat(dir_loc2,os_slash,A2(image_num).name));
    X2 = imadjust(X2);
    if( strcmpi(img_format,'*.dcm') )%image is in dcm format and not segmented.  
        X = dicomread(strcat(dir_loc,os_slash,A(image_num).name));%read dicom images. 
        X = imadjust(X);%adjust image contrast.
        X_cropped = imcrop(X,crop_rect);%crop image to extract ROI.
        BW=X;
        threshold_value = 0.90*max(BW(:));
        BW(BW<threshold_value)=false;BW(BW>=threshold_value)=true;
        BW= logical(BW);
        BW(round(0.94*size(BW,1)):end,:)=false;
        BW=imclearborder(BW);
        BW = bwareaopen(BW,5);  
    else%image is not in dcm format
        X = imread(strcat(dir_loc,os_slash,A(image_num).name));%read if image is in tif or tiff format. 
        X = rgb2gray(X);
        BW=X;
        level = graythresh(BW);
        BW = im2bw(BW,level);
        BW= logical(BW);
        BW(round(0.92*size(BW,1)):end,:)=false;
        BW(round(1:0.08*size(BW,1)),:)=false;
        BW(:,round(0.92*size(BW,1)):end)=false;
        BW(:,round(1:0.08*size(BW,1)))=false;
        BW = bwareaopen(BW,12);       
    end
        imwrite(BW,strcat('mask',num2str(image_num),'.tiff'),'tiff')
        bw = uint8(BW);
        BW_skel = bwmorph(BW,'skel',Inf);
        BW_skel(round(0.90*size(BW_skel,1)):end,:)=false;
        BW_skel(round(1:0.15*size(BW_skel,1)),:)=false;
        BW_skel(:,round(0.90*size(BW_skel,1)):end)=false;
        BW_skel(:,round(1:0.15*size(BW_skel,1)))=false;
        BW_skel = bwareaopen(BW_skel,9);
        bw_skel = uint8(BW_skel);
%%%write gray pixel values to mask
        for dim1=1:size(bw,1)
            for dim2 = 1:size(bw,2)             
                if bw(dim1,dim2)==1 && bw_skel(dim1,dim2)==1 
                    bw(dim1,dim2) = X(dim1,dim2);
                    bw_skel(dim1,dim2) = bw(dim1,dim2);
                end%end if           
            end%end dim2
        end%end dim1
        slice(:,:,image_num)=BW;  
        slice_original(:,:,image_num)=X;
        
end%image_num
 
%% MAKE PIPE AROUND ROI
 n=size(A,1);
 center_x_mtrx = round(n/2);
 center_y_mtrx = round(n/2);
 roi_radius = round(0.35*n);%pipe radius.
 crop_rect = [100,100,410,410];%determined by trial and error.
 roi_radius_sq = round(0.35*crop_rect(4));
 tiff_image_masked = zeros(n,n);
%  pipe_3D = zeros(size(slice,1)+1,size(slice,2)+1,2*size(slice,3));
 pipe_3D = zeros(n+1,n+1,n+2);
 super_imposed = zeros(size(slice));
 inscribing_3D = zeros(size(slice));
 x_min=center_x_mtrx - roi_radius;
 y_min=center_y_mtrx - roi_radius;
 x_max=center_x_mtrx + roi_radius;
 y_max=center_y_mtrx + roi_radius;

 for dim3 = 1:round(1.4*n)
%      for dim3 = 1:size(slice,3)
     for i = 1:n+1;
        
       difference_from_center_x = (center_x_mtrx - i);
            
       for j = 1:n+1;
           
          difference_from_center_y = (center_y_mtrx - j);
        
             distance_from_center_to_voxel = sqrt( difference_from_center_x * difference_from_center_x...
                 + difference_from_center_y * difference_from_center_y );
               
             if(distance_from_center_to_voxel >= roi_radius)  
                 
                 %create an artifical wall (pipe) around scaffold
                 tiff_image_masked(i,j) = white_intensity_value;
          
             end%if
             if(distance_from_center_to_voxel >= roi_radius_sq)  
                 
                 %create an artifical wall (pipe) around scaffold
                 tiff_image_sq(i,j) = white_intensity_value;
          
             end%if
       end%j
    end%i;
    pipe_3D(:,:,dim3) =tiff_image_masked;
 end
 
% img_combined = pipe_3D + slice;
% slice_reduced = slice(:,:,begin:stop);
% pipe_square = repmat(inscribing_square,1,1,size(pipe_3D,3));
% pipe_square_flipped =permute(pipe_square,[3 2 1]);
slice = slice(:,:,begin:stop);
S_mean = mean(slice,3);
S_mean(S_mean~=0)=1; 
S_mean(S_mean==0)=0;

%%with an even smaller sq.
crop_rect1 = [120,155,260,230];%crop_rect1 = [120,155,260,195];
slice_sq1 = slice(round(1*(crop_rect1(2)+1)):round((crop_rect1(2)+crop_rect1(4))),...
    crop_rect1(1)+1:round(1*(crop_rect1(1)+crop_rect1(3))),:);
if mod(size(slice_sq1,2),2)==0 && mod(size(slice_sq1,3),2)==0
    slice_sq1 = slice_sq1(1:end-1,1:end-1);
end  

%pad third dimension with zeros to make 3D square array.
slice = cat(3,slice,zeros(size(slice,1),size(slice,2),size(slice,2)-size(slice,3)));
% % slice_flipped = permute(slice,[3 2 1]);
% % slice_ALIGNED = slice_flipped;
slice_ALIGNED = slice;

slice_ALIGNED = cat(3,zeros(size(slice_ALIGNED,1),size(slice_ALIGNED,2),...
    size(pipe_3D,1)-size(slice_ALIGNED,3)),slice_ALIGNED);
slice_ALIGNED = cat(1,zeros(size(pipe_3D,3)-size(slice_ALIGNED,1),size(slice_ALIGNED,2),...
    size(slice_ALIGNED,3)),slice_ALIGNED);
slice_ALIGNED = cat(2,zeros(size(slice_ALIGNED,1),size(pipe_3D,2)-size(slice_ALIGNED,2),...
    size(slice_ALIGNED,3)),slice_ALIGNED);
slice1 = slice_ALIGNED;

%using largest square x-section.crop_rect = [125,104,266,256];
alpha = 0.97;
slice_sq=slice(round(0.8*(crop_rect(2)+1)):round((crop_rect(2)+crop_rect(4))),...
    crop_rect(1)+1:round(1*(crop_rect(1)+crop_rect(3))),:);
% crop_rect1 = [120,155,260,195];
% slice_sq1 = slice(round(1*(crop_rect1(2)+1)):round((crop_rect1(2)+crop_rect1(4))),...
%     crop_rect1(1)+1:round(1*(crop_rect1(1)+crop_rect1(3))),:);
k1 = size(slice_sq,1);
k2 = size(slice_sq,2);
slice_sq = cat(1,zeros(size(slice_sq,3)-size(slice_sq,1),size(slice_sq,2),...
    size(slice_sq,3)),slice_sq);
slice_sq=slice_sq(:,:,1:k1);
slice_sq = cat(2,zeros(size(slice_sq,1),1,size(slice_sq,3)),slice_sq);
slice_sq = cat(3,zeros(size(slice_sq,1),size(slice_sq,2),1),slice_sq);
% tiff_image_sq = tiff_image_sq(round(alpha*(0.5*(size(slice,1))-roi_radius_sq)):...
%     end-alpha*(size(slice,1)-(2*roi_radius_sq+(0.5*size(slice,1))-roi_radius_sq)),:);
% tiff_image_sq = tiff_image_sq(:,round(alpha*(0.5*(size(slice,1))-roi_radius_sq)):...
%     end-alpha*(size(slice,1)-(2*roi_radius_sq+(0.5*size(slice,1))-roi_radius_sq)));
tiff_image_sq = tiff_image_sq(1:size(slice_sq,3),1:size(slice_sq,2));
if mod(size(tiff_image_sq,1),2)==0 && mod(size(tiff_image_sq,2),2)==0
    tiff_image_sq = tiff_image_sq(1:end-1,1:end-1);
end    
pipe_sq = repmat(tiff_image_sq,1,1,size(slice_sq,1));
pipe_sq_flipped = permute(pipe_sq,[3 2 1]);
slice_sq_flp = cat(3,zeros(size(slice_sq,1),size(slice_sq,2),...
    size(pipe_sq,1)-size(slice_sq,3)),slice_sq);
slice_sq_flp = cat(2,zeros(size(slice_sq_flp,1),round(0.5*(size(pipe_sq,2)-...
    size(slice_sq_flp,2))),size(slice_sq_flp,3)),slice_sq_flp);
slice_sq_flp = cat(2,slice_sq_flp,zeros(size(slice_sq_flp,1),...
    round(0.5*(size(pipe_sq,2)-size(slice_sq_flp,2))),size(slice_sq_flp,3)));
slice_sq_flp = cat(2,slice_sq_flp,zeros(size(slice_sq_flp,1),...
    size(pipe_sq_flipped,2)-size(slice_sq_flp,2),size(slice_sq_flp,3)));
% S_inscribed = pipe_sq_flipped + slice_sq_flp;
% Q1 = ones(size(slice_sq1));
% Q1(Q1==1) = white_intensity_value;
% x1=(1:k1+1);x2=(1:k2+1);
% tiff_image_sq = (x1 - 0.5*k1).^2 + (x2 - 0.5*k2).^2 >=(roi_radius_sq)^2;

%using full scaffold.
S = permute(pipe_3D,[3 2 1]);
S_combined = S+slice1;
% S_inscribed = pipe_square_flipped+slice1;
pipe_ex = repmat(tiff_image_masked,1,1,2);
pipe_ex1 = permute(pipe_ex,[3 2 1]);
% pipe_square_ex = repmat(inscribing_square,1,1,2);
% pipe_square_ex1 = permute(pipe_square_ex,[3 2 1]);
% S_inscribed1 = cat(1,S_inscribed,pipe_square_ex1);
S_combined1 = cat(1,S_combined,pipe_ex1);
S_sq = slice_sq1(1:end,101:end,185:end-65);
if mod(size(S_sq,3),2)==0 
    S_sq = S_sq(1:end-1,:,1:end-1);
end 
S_sq = S_sq(:,1:size(S_sq,3),:);
% S_longer = S_sq;

S_longer = S_combined;
reconstruction_fid_flip = fopen('3d_reconstruction1.txt','w');
global_walls_fid = fopen('global_walls1.txt','w');
fprintf(reconstruction_fid_flip,'ZONE i=%6i,j=%6i,k=%6i\n',...
        size(S_longer,1),size(S_longer,2),size(S_longer,3));  
    
%move all mask into a newly created directory
if ~exist('scaffold_mask','dir')
    mkdir('scaffold_mask');     
end
movefile('*.tiff','scaffold_mask');

for slice_number = 1:size(S_longer,3)
    for j = 1:size(S_longer,2)            
            for i = 1:size(S_longer,1)
                if  S_longer(i,j,slice_number)~=0;              
                    fprintf(global_walls_fid,'%6i %6i %6i %c\n',i,j,slice_number,'T' ); 
                else
                    fprintf(global_walls_fid,'%6i %6i %6i %c\n',i,j,slice_number,'F' );
                end
                fprintf(reconstruction_fid_flip,'%6i %6i %6i %6i\n',...
                    i,j,slice_number,S_longer(i,j,slice_number));
            end; %i        
    end; %j
 end%slice_number 
 
%close files
fclose(reconstruction_fid_flip);
fclose(global_walls_fid); 

%send email if script fails.
email_address = 'ok26@njit.edu';
subject = 'scipt success';
message = 'script ran successfully';
if slice_number==size(S_longer,3)
    setpref('Internet','E_mail',email_address);
    setpref('Internet','SMTP_Server','mail');
    sendmail(email_address,subject,message)
end

% % % slice1 = permute(slice_ALIGNED,[1 3 2]);
% % % sum_1=0;sum_1_pipe=0;
% % % sum_2=0;sum_2_pipe=0;
% % % sum_3=0;sum_3_pipe=0;
% % % for kk = 1:size(slice,3)
% % %     for jj =1:size(slice,2)
% % %         for ii = 1:size(slice,1)
% % %            sum_1 = sum_1 + ii*slice(ii,jj,kk) ;
% % %            sum_2 = sum_2 + jj*slice(ii,jj,kk) ;
% % %            sum_3 = sum_3 + kk*slice(ii,jj,kk) ;
% % %            sum_1_pipe = sum_1_pipe + ii*pipe_3D(ii,jj,kk);
% % %            sum_2_pipe = sum_2_pipe + jj*pipe_3D(ii,jj,kk);
% % %            sum_3_pipe = sum_3_pipe + kk*pipe_3D(ii,jj,kk); 
% % %         end
% % %     end
% % % end
% % % centroid_1 = sum_1/sum(slice(:));
% % % centroid_2 = sum_2/sum(slice(:));
% % % centroid_3 = sum_3/sum(slice(:));
% % % pipe_cen_sum=pipe_3D(1:size(slice,1),1:size(slice,2),1:size(slice,3));
% % % pipe_1_centroid = sum_1_pipe/sum(pipe_cen_sum(:)); 
% % % pipe_2_centroid = sum_2_pipe/sum(pipe_cen_sum(:));
% % % pipe_3_centroid = sum_3_pipe/sum(pipe_cen_sum(:));
% % % disp1 = pipe_1_centroid - centroid_1 ;
% % % disp2 = pipe_2_centroid - centroid_2 ;
% % % disp3 = pipe_3_centroid - centroid_3 ;
% % % slice_ALIGNED1 = imtranslate(slice,[disp1,disp2,disp3]);
% % % slice_ALIGNED2 = imtranslate(slice,[disp1,disp2,0]);
% % % slice_ALIGNED = slice_ALIGNED2;
% slice_ALIGNED = cat(1,zeros(size(pipe_3D,3)-size(slice_ALIGNED,1),size(slice_ALIGNED,2),...
% size(slice_ALIGNED,3)),slice_ALIGNED);
% slice1 = slice_ALIGNED;

% Turn off this warning "Warning: Image is too big to fit on screen; displaying at 33% "
% To set the warning state, you must first know the message identifier for the one warning you want to enable. 
% Query the last warning to acquire the identifier.  For example: 
% warnStruct = warning('query', 'last');
% msgid_integerCat = warnStruct.identifier
% msgid_integerCat =
%    MATLAB:concatenation:integerInteraction
% warning('off', 'Images:initSize:adjustingMag');
% ['''-s 10''']
%   inscribing_square = zeros(n+1,n+1);
%  inscribing_square(1:round(0.34*n),:) = true;
%  inscribing_square(round(0.72*n):end,:) = true;
%  inscribing_square(:,1:round(0.34*n)) = true;
%  inscribing_square(:,round(0.72*n):end) = true;
%  inscribing_square = ~inscribing_square;
%  inscribing_square(round(0.34*n)+4:round(0.72*n)-4,...
%      round(0.34*n)+4:round(0.72*n)-4) = false;
%  [Q,rec1]=imcrop(slice(:,:,250));