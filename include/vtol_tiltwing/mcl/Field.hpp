#ifndef VTOL_TILTWING_FIELD_HPP
#define VTOL_TILTWING_FIELD_HPP

#include <vtol_tiltwing/mcl/FieldBase.hpp>
#include <string>
#include <utility>

template <class T>
class Field : public virtual FieldBase {
private:
    const std::string _id;
    T _val;
    float _time;
public:
    Field(std::string id)
            : _id(std::move(id)),
              _time(-1) {}

    Field(std::string id, T val)  // Instead of const std::string &id, passing by value and using std::move()
            : _id(std::move(id)),
              _val(val),
              _time(-1) {}

    std::string getId(){ return _id; }
    T getVal(){ return _val; }
    float getTime() { return _time; }
    void setVal(T val){ _val = val; }
};

#endif //VTOL_TILTWING_FIELD_HPP
