#ifndef VTOL_TILTWING_FIELDBASE_HPP
#define VTOL_TILTWING_FIELDBASE_HPP

/**
 * @brief Dummy class so that we can create pointers of type StateFieldBase that point to objects of
 * type StateField<T>. See "StateField.hpp"
 */
class FieldBase{
public:
    virtual ~FieldBase() {};
};

#endif //VTOL_TILTWING_FIELDBASE_HPP
