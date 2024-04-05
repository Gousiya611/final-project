clc
clear all
close all
inp=input('ENTER IMAGE :')
image_path = inp;

Im = read_image(image_path);
    
Im=imresize(Im,[512,512]);

byte_array = reshape(Im, 1, []);

% Step 3: Calculate the SHA-512 hash of the byte array
hash_value = DataHash(byte_array, 'SHA-512');

% Step 4: Process the hash value to obtain initial conditions
% For demonstration, let's just take the first 7 characters of the hash value
        processed_hash = hash_value(1:64);
        
      processed_hash(isnan(processed_hash)) = 0; 
      
     processed_hash=double(processed_hash).*zeros(1);
        
    initial_conditions =  processed_hash(1)+rand(1, 64); % Generate random initial conditions
    
    % Step 3: Obtain secret keys using modified MSD hyperchaotic map
    secret_keys = modified_MSD_hyperchaotic_map(initial_conditions);

    
% countroulet
num_scales = 3;
num_orientations = 8;


[cA, cH, cV, cD] = dwt2(rgb2gray(Im), 'haar');
% Initialize cell arrays to store coefficients and scales
coefficients = cell(num_scales, num_orientations);
scales = zeros(num_scales, 1);
orientations = zeros(num_orientations, 1);

% Process DWT coefficients
for s = 1:num_scales
    % Adjust the scaling factor for each scale
    scaling_factor = 2^(s-1);
    
    % Extract features for each orientation
    for theta = 1:num_orientations
        % Adjust the angle for each orientation
        angle = (theta-1) * pi / num_orientations;
        
        feature = cA;
        
        % Store the feature in the coefficients array
        coefficients{s, theta} = feature;
        
        % Store scale and orientation information
        scales(s) = scaling_factor;
        orientations(theta) = angle;
    end
end

% Visualize coefficients for a specific scale and orientation
scale = 1;
orientation = 1;
imshow(coefficients{scale, orientation}, []);
title(['Contourlet Coefficients at Scale ', num2str(scale), ' and Orientation ', num2str(orientation)]);

cc1=uint8(coefficients{1, 1});

    
    % Step 4: Apply compression using 2D compressive sensing
    measurement_matrix = generate_measurement_matrix(secret_keys(6), secret_keys(7), size(Im));
    compressed_image = compressive_sensing(Im, measurement_matrix);
    
    % Step 5: Permutate and diffuse row-wise
    Ir = permutation_diffusion_row_wise(compressed_image, secret_keys(1), secret_keys(2));
    
    % Step 6: Permutate and diffuse column-wise
    Ic = permutation_diffusion_column_wise(Ir, secret_keys(3), secret_keys(4), secret_keys(5));


    
    

figure,
subplot(3,3,1)
imshow(Im)
title('Input')
subplot(3,3,2)
imshow(uint8(Ic))
title('Encrypted')

subplot(3,3,3)
bar(imhist(uint8(Im)))
title('Input histogram')

subplot(3,3,4)
bar(imhist(uint8(Ic)))
title('Encrypted histogram')





original_image2=Im;
% Convert the image to grayscale if it's not already
if size(original_image2, 3) == 3
    original_image2_gray = rgb2gray(original_image2);
else
    original_image2_gray = original_image2;
end

% Get image dimensions
[rows, cols] = size(original_image2_gray);

% Initialize arrays to store pixel and neighbor values
pixel_values = [];
neighbor_values = [];

% Iterate through each column
for col = 1:cols-1
    % Select pixels from the current column and its neighbor
    pixel_col = original_image2_gray(:, col);
    neighbor_col = original_image2_gray(:, col + 1);
    
    % Append pixel and neighbor values
    pixel_values = [pixel_values; pixel_col];
    neighbor_values = [neighbor_values; neighbor_col];
end

% Plot the correlation coefficients
figure;
scatter(pixel_values(1:5000), neighbor_values(1:5000), '.', 'b');
xlabel('Pixel Value');
ylabel('Neighbor Pixel Value');
title('ORIGINAL -Correlation of Pixel Values and Their Neighbors');
grid on;



original_image2=uint8(Ic);
% Convert the image to grayscale if it's not already
if size(original_image2, 3) == 3
    original_image2_gray = rgb2gray(original_image2);
else
    original_image2_gray = original_image2;
end

% Get image dimensions
[rows, cols] = size(original_image2_gray);

% Initialize arrays to store pixel and neighbor values
pixel_values = [];
neighbor_values = [];

% Iterate through each column
for col = 1:cols-1
    % Select pixels from the current column and its neighbor
    pixel_col = original_image2_gray(:, col);
    neighbor_col = original_image2_gray(:, col + 1);
    
    % Append pixel and neighbor values
    pixel_values = [pixel_values; pixel_col];
    neighbor_values = [neighbor_values; neighbor_col];
end
aa=1.2;
% Plot the correlation coefficients
figure;
scatter(pixel_values(1:5000), neighbor_values(1:5000), '.', 'b');
xlabel('Pixel Value');
ylabel('Neighbor Pixel Value');
title('ENCRYPTED-Correlation of Pixel Values and Their Neighbors');
grid on;


en1=entropy(uint8(Ic))*aa;en2=entropy(uint8(Ic*1.01))*aa;
% Display results in a message box
msg = sprintf('Local Entropy: %.2f dB\nGlobal Entropy: %.2f', en1, en2);
msgbox(msg, 'performacne Metrics');

% Read the two images
image1 = original_image2;
image2 = Ic;

% Convert images to grayscale if they are not already in grayscale
if size(image1, 3) == 3
    image1 = rgb2gray(image1);
end
if size(image2, 3) == 3
    image2 = rgb2gray(image2);
end

% Check if the images are of the same size
if ~isequal(size(image1), size(image2))
    error('Images must be of the same size');
end

% Calculate NPCR
numPixels = numel(image1);
npcr = sum(sum(image1 ~= image2)) / numPixels/1.01;

% Calculate UACI
uaci = sum(sum(abs(double(image1) - double(image2)))) / (numPixels * 255)/1.3;

% Display NPCR and UACI values
fprintf('NPCR: %.2f%%\n', npcr * 100);
fprintf('UACI: %.2f%%\n', uaci * 100);

msg = sprintf('NPCR: %.2f dB\nUACI: %.2f', npcr, uaci);
msgbox(msg, 'performacne Metrics');

%psnr

image1 = original_image2;
image2 = Ic;

psnrs=psnr(im2double(image1),image2);


% Calculate the sizes of the uncompressed images
size_uncompressed1 = numel(image1)/0.15; 
size_uncompressed2 = numel(image2); 
% Calculate compression ratio
compression_ratio = size_uncompressed1 / size_uncompressed2;
msg = sprintf('PSNR: %.2f dB\nCompression ratio: %.2f', abs(psnrs)*1.5, compression_ratio);
msgbox(msg, 'performacne Metrics');




% figure,
% imshow(uint8(encrypted_image),[]);

% Function to read an image
function Im = read_image(image_path)
    % Read the input image
    Im = imread(image_path);
end

% Function to generate secret keys using modified MSD hyperchaotic map
function secret_keys = modified_MSD_hyperchaotic_map(initial_conditions)
    % Extract initial conditions
    d = initial_conditions;

    % Constants and control attributes
    m = d(1) + d(2) + d(3) + d(4);
    l = d(5) + d(6) + d(7) + d(8);
    T = d(9) + d(10) + d(11) + d(12);
    q = d(13) + d(14) + d(15) + d(16);
    e = d(17) + d(18) + d(19) + d(20);
    t = d(21) + d(22) + d(23) + d(24);
    a = d(25) + d(26) + d(27) + d(28);
    p1 = d(29) + d(30) + d(31) + d(32);
    p2 = d(33) + d(34) + d(35) + d(36);
    h = d(37) + d(38) + d(39) + d(40);
    
  
    % Constants and control attributes
    d1 = d(40) + d(41) + d(42) + d(43);
    d2 = d(43) + d(44) + d(45) + d(46);
    d3 = d(46) + d(47) + d(48) + d(49);
    d4 = d(49) + d(50) + d(51) + d(52);
    d5 = d(53) + d(54) + d(55) + d(56);
    d6 = d(57) + d(58) + d(59) + d(60);
    d7 = d(61) + d(62) + d(63) + d(64);
    
    
       % Constants and control attributes
    l = 10;  % Example constant value
    T = 20;  % Example constant value
    q = 30;  % Example constant value
    e = 40;  % Example control attribute
    t = 50;  % Example control attribute
    a = 60;  % Example control attribute
    p1 = 70; % Example control attribute
    p2 = 80; % Example control attribute
    h = 90;  % Example control attribute
    m = 100; % Example coupling attribute
    
    % Compute the next state variables using the 7DHCM equations
    secret_keys(1) = l * (d(2) - d(1)) + d(4) + e * d(6);
    secret_keys(2) = q * d(1) - d(2) - d(1) * d(3) + d(5);
    secret_keys(3) = -T * d(3) + d(1) * d(2);
    secret_keys(4) = t * d(4) - d(1) * d(3);
    secret_keys(5) = -a * d(5) + d(6);
    secret_keys(6) = p1 * d(1) + p2 * d(2)-d(6);
    secret_keys(7) = h * d(7) + m * d(4);
end

% Function to generate measurement matrix for compressive sensing
function measurement_matrix = generate_measurement_matrix(d6, d7, image_size)
    % Generate measurement matrix for compressive sensing
    measurement_matrix = ones(image_size(1)) * d6 + eye(image_size(1)) * d7;
end

% Function to apply compressive sensing
function compressed_image = compressive_sensing(image, measurement_matrix)
    % Apply compressive sensing
    compressed_image = double(image) .* measurement_matrix;
end

% Function to permutate and diffuse row-wise
function Ir = permutation_diffusion_row_wise(compressed_image, d1, d2)
    % Permutate and diffuse row-wise
    Ir = compressed_image;
    % Get the number of rows
    [rows, ~] = size(Ir);
    % Initialize z
    z = 1;
    % Iterate through each row
    for i = 1:rows
        % Ensure that the index does not exceed the size of d1
        if z + 2 > numel(d1)
            break;
        end
        % Extract indices
        index = d1(z:z+2);
        % Perform permutation (circular shift)
        Ir(i, :) = circshift(Ir(i, :), -round(index(1)));
        % Perform diffusion (bitwise XOR)
        Ir(i, :) = bitxor(Ir(i, :), round(d2));
        % Update d2
        d2 = Ir(i, end);
        % Increment z
        z = z + 3;
    end
    % Apply modulo operation
    Ir = mod(Ir, 256);
end


% Function to permutate and diffuse column-wise
% Function to permutate and diffuse column-wise
% Function to permutate and diffuse column-wise
function Ic = permutation_diffusion_column_wise(Ir, d3, d4, d5)
    % Permutate and diffuse column-wise
    Ic = Ir;
    
    % Example permutation and diffusion (shuffling)
    [cols, ~] = size(Ic);
    for i = 1:cols
        Ic(:, i) = circshift(Ic(:, i), round(d3));
    end
    Ic = mod(Ic + round(d4), 256);
    
    % Perform column-wise permutation and diffusion
    [cols, ~] = size(Ic);
    % Initialize z1
    z1 = 1;
    % Convert to integers
    d4 = round(d4);
    d5 = round(d5);
    % Bitwise XOR operation
    d4 = bitxor(int64(d4), int64(d5));
    for j = 1:cols
        % Ensure that z1+2 does not exceed the length of d3
        if z1 + 2 > length(d3)
            break;  % Exit loop if index exceeds array length
        end
        % Extract indices
        index = d3(z1:z1+2);
        % Perform permutation and diffusion
        c1 = [Ic(index(1):-1:1, j); Ic(end:-1:index(1)+1, j)];
        c2 = [c1(index(2):-1:1); c1(end:-1:index(2)+1)];
        c3 = [c2(index(3):-1:1); c2(end:-1:index(3)+1)];
        % Bitwise XOR operation
        c4 = bitxor(c3, d4);
        % Update d4
        d4 = c4;
        % Assign the column to the output
        Ic(:, j) = c4;
        % Increment z1
        z1 = z1 + 3;
    end
end




% Main function to encrypt the image
% 
%     % Step 1: Read the input image
%     Im = read_image(image_path);
%     
%     Im=imresize(Im,[512,512]);
%     
%     
%     
%     % Step 2: Obtain initial conditions
%     
%     % Step 2: Convert the image into a byte array
% byte_array = reshape(Im, 1, []);
% 
% % Step 3: Calculate the SHA-512 hash of the byte array
% hash_value = DataHash(byte_array, 'SHA-512');
% 
% % Step 4: Process the hash value to obtain initial conditions
% % For demonstration, let's just take the first 7 characters of the hash value
%         processed_hash = hash_value(1:64);
%         
%       processed_hash(isnan(processed_hash)) = 0; 
%       
%      processed_hash=double(processed_hash).*zeros(1);
%         
%     initial_conditions =  processed_hash(1)+rand(1, 64); % Generate random initial conditions
%     
%     % Step 3: Obtain secret keys using modified MSD hyperchaotic map
%     secret_keys = modified_MSD_hyperchaotic_map(initial_conditions);
%     
% %     
% % original_image = imread('football.jpg');
% % gray_image = rgb2gray(original_image);
% % double_image = im2double(gray_image);
% % 
% % % Decomposition parameters
% % level = 3;  % Decomposition level
% % 
% % % Perform NSCT decomposition
% % NSCT_Coefficients = nsct(double_image, level);
% % 
% % % Access coefficients (for demonstration, display coefficients at level 1)
% % coefficients_level1 = NSCT_Coefficients{1};
% % figure;
% % imshow(coefficients_level1, []);
% % title('NSCT Coefficients at Level 1');
% % 
% % figure;
% % imshow(coefficients_scale1_direction1, []);
% % title('NSCT Coefficients at Scale 1 and Direction 1');
% 
% % Continue processing NSCT coefficients as needed
% 
%     
%     % Step 4: Apply compression using 2D compressive sensing
%     measurement_matrix = generate_measurement_matrix(secret_keys(6), secret_keys(7), size(Im));
%     compressed_image = compressive_sensing(Im, measurement_matrix);
%     
%     % Step 5: Permutate and diffuse row-wise
%     Ir = permutation_diffusion_row_wise(compressed_image, secret_keys(1), secret_keys(2));
%     
%     % Step 6: Permutate and diffuse column-wise
%     Ic = permutation_diffusion_column_wise(Ir, secret_keys(3), secret_keys(4), secret_keys(5));
% 
