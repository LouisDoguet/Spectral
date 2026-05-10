#ifndef IDXREADER_H
#define IDXREADER_H

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdint>


inline std::string toHex(unsigned char &num){
	std::stringstream ss;
	ss << "0x" << std::uppercase << std::setfill('0') << std::setw(2)
	   << std::hex << static_cast<int>(num) << " ";
	return ss.str();
}

inline std::string toHex(int &num) {
	std::stringstream ss;
	ss << "0x" << std::uppercase << std::setfill('0') << std::setw(2) << std::hex << num << " ";
	return ss.str();
}


inline unsigned int read_uint32(std::ifstream &file, unsigned int shift){
	file.seekg(shift*4);
	unsigned char b0 = file.get();
	unsigned char b1 = file.get();
	unsigned char b2 = file.get();
	unsigned char b3 = file.get();

	return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

// Stores image data from the dataset
class Image {
	public:

	Image() = default;
	Image(std::vector<unsigned char>&& pixels, int nrow, int ncol) :
		pixels(std::move(pixels)),
		nrow(nrow),
		ncol(ncol),
		norm_pixels(normalize()) {};

	std::vector<unsigned int> toUint() const {
		std::vector<unsigned int> result(ncol*nrow);
		for (int i=0 ; i < (int) pixels.size() ; i++)
			result[i] = static_cast<unsigned int>(pixels[i]);
		return result;
	}

	// Getters
	const std::vector<unsigned char>& getPixels() const {return pixels;}
	float getPixel(int n) const {return norm_pixels.at(n);}
	int size() const {return (nrow*ncol);}

	// Friends
	friend std::ostream& operator<<(std::ostream&, Image&);

	private:
	const std::vector<unsigned char> pixels;
	const int nrow;
	const int ncol;
	const std::vector<float> norm_pixels;

	// Normalize pixel values to [0,1]
	std::vector<float> normalize() const {
		std::vector<float> result(pixels.size());
		for (size_t i = 0; i < pixels.size(); ++i)
			result[i] = static_cast<float>(pixels[i]) / 255.0f;
		return result;
	}
};


// Database class :
// Reads idx file to get the value of the images stored in the dataset
class idxDatabase {
	public:

	idxDatabase() = default;
	idxDatabase(const std::string &database_path): path(database_path), file(database_path, std::ios::binary)
	{ read_header(); };

	// Getters
	std::string getPath() const {return path;}
	int mNumber() const {return magic_number;}
	int nImage() const {return number_of_image;}
	int nRows() const {return rows;}
	int nCols() const {return cols;}

	// Read the image at position 'id' in the file
	Image read_image(int id){
		int total_shift = 16;
		total_shift+= ( id * rows  * cols );

		file.seekg(total_shift);
		std::vector<unsigned char> pixels(rows*cols);
		file.read(reinterpret_cast<char*>(pixels.data()), rows*cols);

		return Image(std::move(pixels),rows,cols);
	}

	private:
	const std::string path;
	std::ifstream file;
	int magic_number;
	int number_of_image;
	int rows;
	int cols;

	void read_header() {
		magic_number = read_uint32(file, 0);
		number_of_image = read_uint32(file, 1);
		rows = read_uint32(file, 2);
		cols = read_uint32(file, 3);
	}

};

// Reads idx1 label file
class idxLabelDB {
public:
	idxLabelDB() = default;
	idxLabelDB(const std::string& path) : path(path), file(path, std::ios::binary) {
		magic = read_uint32(file, 0);
		count = read_uint32(file, 1);
	}

	int getLabel(int id) {
		file.seekg(8 + id);
		return static_cast<int>(static_cast<unsigned char>(file.get()));
	}

	int nLabels() const { return count; }

private:
	std::string path;
	std::ifstream file;
	int magic = 0;
	int count = 0;
};

// Surcharges
inline std::ostream &operator<<(std::ostream &os, idxDatabase &db) {
	os << "Path    :   " << db.getPath() << std::endl
	   << "Mnumber :   " << db.mNumber() << std::endl
	   << "Nimages :   " << db.nImage() << std::endl
	   << "R,C     :   " << db.nRows() << "," << db.nCols() << std::endl;
	return os;
};

inline std::ostream &operator<<(std::ostream &os, Image &img) {

	const std::string shades = " .:-=+*#%@";
	const std::vector<unsigned char>& pixels = img.getPixels();

	for (int i = 0; i < img.size(); ++i) {
		unsigned char pixel = pixels[i];
		int shade_idx = pixel * (shades.size() - 1) / 255;
		os << shades[shade_idx] << shades[shade_idx];
		if ((i + 1) % img.ncol == 0)
		os << std::endl;
	}
	return os;
};

#endif
