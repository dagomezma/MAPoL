// FieldMap.h
// Copyright 2018/8/3 Robin.Rowe@cinepaint.org
// License MIT open source

#ifndef FieldMap_h
#define FieldMap_h

#include <map>

template <class T> 
struct string_view_compare 
{	bool operator() (const T& x, const T& y) const 
	{	return x.compare(y)<0;
}	};

//template <unsigned fieldCount>
class FieldMap
{	std::map<std::string_view,std::string_view,string_view_compare<std::string_view> > field;
	//const char* field[fieldCount] = {};
	char empty;
	char* p;
	const char* StripFieldname(const char* text) const
	{	if(!text)
		{	return &empty;
		}
		const char* sep = strstr(text,": ");
		if(sep)
		{	return sep+2;
		}
		return text;
	}
public:
	FieldMap()
	{	empty = 0;
		p = nullptr;
	}
	void Clear()
	{	p=0;
		field.clear();
	}
	void Set(char* data)
	{	p = data;
	}
#if 0
	bool IsInvalid(unsigned i) const
	{	return fieldCount<=i;
	}
	char* GetFieldData(unsigned i)
	{	if(IsInvalid(i))
		{	return &empty;
		}
		return field[i];
	}
	const char* GetFieldData(unsigned i) const
	{	if(IsInvalid(i))
		{	return &empty;
		}
		return field[i];
	}
	const char* operator[](unsigned i) const
	{	return StripFieldname(GetFieldData(i));
	}

	char* Parse(const char* fieldName,unsigned i)
	{	
#if 0
		if(IsInvalid(i))
		{	return 0;
		}
#endif
		char* found = strstr(p,fieldName);
		if(!found)
		{	return 0;
		}
		field[i] = found;
		p = found + 1;
		char* ends = strchr(p,'\n');
		if(ends)
		{	*ends= 0;
			p = ends + 1;
		}
		return found;
	}
#endif
	bool AdvancePast(const char* text)
	{	char* seek = strstr(p,text);
		if(!seek)
		{	return false;
		}
		p = seek + 2;
		return true;	
	}
	char* SkipWhitespace()
	{	while(isspace(*p))
		{	p++;
		}
		return p;
	}
	char* Find(char c)
	{	return strchr(p,c);
	}
};

#endif
