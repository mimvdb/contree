/*
* Adapted from STreeD
* https://github.com/AlgTUDelft/pystreed
* by Jacobus G. M. van der Linden
*/

#ifndef BITSET_H
#define BITSET_H

#include <algorithm>
#include <cstddef>
#include <cstring>

struct dynamic_bitset {
	using base_type = unsigned long;

	base_type* bitset;
	size_t elements;

	dynamic_bitset(size_t size) {
		elements = (size - 1) / sizeof(base_type) + 1;
		bitset = new base_type[elements];
		std::fill(bitset, bitset + elements, 0);
	}

	dynamic_bitset() : dynamic_bitset(1) {}

	~dynamic_bitset() {
		delete[] bitset;
	}

	dynamic_bitset(const dynamic_bitset& other) : elements(other.elements) {
		bitset = new base_type[elements];
		std::memcpy(bitset, other.bitset, elements * sizeof(base_type));
	}

	dynamic_bitset& operator=(const dynamic_bitset& other) {
		if (this == &other)
			return *this;

		elements = other.elements;
		base_type* new_bitset = new base_type[elements];
		std::memcpy(new_bitset, other.bitset, elements * sizeof(base_type));
		delete[] bitset;
		bitset = new_bitset;
		return *this;
	}

	bool operator==(const dynamic_bitset& other) const {
		for (size_t i = 0; i < elements; i++) {
			if (bitset[i] != other.bitset[i]) return false;
		}
		return true;
	}

	inline bool operator!=(const dynamic_bitset& other) const {
		return !((*this) == other);
	}

	void set_bit(size_t index) {
		size_t element = index / sizeof(base_type);
		size_t bit_index = index % sizeof(base_type);
		bitset[element] |= 1UL << bit_index;
	}

	void clear_bit(size_t index) {
		size_t element = index / sizeof(base_type);
		size_t bit_index = index % sizeof(base_type);
		bitset[element] &= 1UL << bit_index;
	}

	void toggle_bit(size_t index) {
		size_t element = index / sizeof(base_type);
		size_t bit_index = index % sizeof(base_type);
		bitset[element] ^= 1UL << bit_index;
	}

};

template <>
struct std::hash<dynamic_bitset> {

	//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
	size_t operator()(const dynamic_bitset& bitset) const {
		using std::size_t;
		using std::hash;
		size_t seed = 0;
		for (int i = 0; i < bitset.elements; i++) {
			seed ^= hash<dynamic_bitset::base_type>()(bitset.bitset[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}

};

#endif // BITSET_H